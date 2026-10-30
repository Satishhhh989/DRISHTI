"""DRISHTI - Real Export System Tests
Phase 09: Interaction, Persistence & Export
"""

import io
import json
import zipfile
import pytest
import trimesh
from fastapi.testclient import TestClient

from app.assets.manager import asset_manager
from app.export.engine import export_engine
from app.main import app
from app.pipeline.reconstructor import export_mesh_glb
from app.project.manager import project_manager
from app.scene.manager import scene_manager
from app.schemas.assets import AssetDimensions, AssetMetadata
from app.schemas.common import Transform, Vector3D, EulerRotation
from app.schemas.scene import (
    GeometryMetadata,
    HumanReconstructionData,
    MeshBounds,
    SceneEntity,
)


@pytest.fixture
def mock_mesh_scene():
    # 1. Register a test photo asset
    photo_asset = AssetMetadata(
        asset_id="asset-photo-export",
        category="original_photo",
        filename="export_photo.jpg",
        relative_path="photos/export_photo.jpg",
        mime_type="image/jpeg",
        size_bytes=4096,
        checksum_sha256="c" * 64,
        dimensions=AssetDimensions(width=1000, height=800),
    )
    scene = scene_manager.create_scene_from_photo(photo_asset, name="Export Test Scene")

    # 2. Create and save a genuine test GLB mesh
    box = trimesh.creation.box(extents=[0.5, 1.7, 0.3])
    glb_bytes = export_mesh_glb(box.vertices, box.faces, name="test_human")
    mesh_asset = asset_manager.register_mesh(
        scene_id=scene.id,
        content=glb_bytes,
        filename="person_0_mesh.glb",
    )

    # 3. Add person with reconstructed mesh
    e0 = SceneEntity(
        id="person-0",
        source_image_id=photo_asset.asset_id,
        name="Person Zero",
        reconstruction_status="ready",
        transform=Transform(
            position=Vector3D(x=-0.5, y=0.0, z=-1.5),
            rotation=EulerRotation(x=0.0, y=0.2, z=0.0),
            scale=Vector3D(x=1.0, y=1.0, z=1.0),
        ),
        reconstruction=HumanReconstructionData(
            status="ready",
            mesh_asset_id=mesh_asset.asset_id,
            mesh_url=f"/api/v1/assets/{mesh_asset.asset_id}/file",
            geometry_metadata=GeometryMetadata(
                vertex_count=len(box.vertices),
                face_count=len(box.faces),
                bounds=MeshBounds(min=[-0.25, -0.85, -0.15], max=[0.25, 0.85, 0.15]),
                dimensions=[0.5, 1.7, 0.3],
            ),
        ),
    )

    # 4. Add person WITHOUT reconstructed mesh (unreconstructed)
    e1 = SceneEntity(
        id="person-1",
        source_image_id=photo_asset.asset_id,
        name="Person One (Unreconstructed)",
        reconstruction_status="unstarted",
    )

    scene.entities = [e0, e1]
    scene_manager.update_scene(scene)
    return scene, e0, e1


client = TestClient(app)


def test_export_single_entity_mesh_glb(mock_mesh_scene):
    scene, e0, _ = mock_mesh_scene
    res = client.get(f"/api/v1/export/mesh/{scene.id}/{e0.id}")
    assert res.status_code == 200
    assert res.headers["content-type"] == "model/gltf-binary"
    assert "person-0_mesh.glb" in res.headers["content-disposition"]

    # Verify bytes load as a valid GLB
    loaded = trimesh.load(trimesh.util.wrap_as_stream(res.content), file_type="glb")
    assert loaded is not None
    if isinstance(loaded, trimesh.Scene):
        assert len(loaded.geometry) > 0
    else:
        assert len(loaded.vertices) > 0


def test_reject_export_unreconstructed_person(mock_mesh_scene):
    scene, _, e1 = mock_mesh_scene
    res = client.get(f"/api/v1/export/mesh/{scene.id}/{e1.id}")
    assert res.status_code == 422
    assert "has no reconstructed 3D mesh" in res.json()["error"]["message"]


def test_export_assembled_scene_glb(mock_mesh_scene):
    scene, _, _ = mock_mesh_scene
    res = client.get(f"/api/v1/export/scene/{scene.id}")
    assert res.status_code == 200
    assert res.headers["content-type"] == "model/gltf-binary"
    assert "assembled_scene" in res.headers["content-disposition"]

    loaded = trimesh.load(trimesh.util.wrap_as_stream(res.content), file_type="glb")
    assert isinstance(loaded, trimesh.Scene)
    assert len(loaded.geometry) >= 1


def test_export_scene_metadata(mock_mesh_scene):
    scene, _, _ = mock_mesh_scene
    res = client.get(f"/api/v1/export/metadata/{scene.id}")
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/json"
    data = res.json()
    assert data["id"] == scene.id
    assert len(data["entities"]) == 2


def test_portable_project_archive_and_safe_import(mock_mesh_scene):
    scene, _, _ = mock_mesh_scene
    project = project_manager.create_project_from_scene(scene, name="Archive Export Project")

    # 1. Export archive
    res = client.get(f"/api/v1/export/project/{project.id}/archive")
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/zip"

    # Verify zip content
    zf = zipfile.ZipFile(io.BytesIO(res.content))
    namelist = zf.namelist()
    assert "project.json" in namelist
    assert "scene.json" in namelist
    assert "manifest.json" in namelist

    # 2. Import archive safely
    files = {"file": ("archive.zip", res.content, "application/zip")}
    data = {"new_name": "Reimported Project"}
    import_res = client.post("/api/v1/projects/import", files=files, data=data)
    assert import_res.status_code == 200
    imp_data = import_res.json()
    assert imp_data["name"] == "Reimported Project"
    assert imp_data["id"].startswith("proj-imp-")


def test_anti_zip_slip_path_traversal_defense():
    # Construct malicious zip with ../../../evil.txt
    malicious_buf = io.BytesIO()
    with zipfile.ZipFile(malicious_buf, "w") as zf:
        zf.writestr("../../../evil.txt", "pwned")
        zf.writestr("project.json", "{}")
        zf.writestr("scene.json", "{}")

    files = {"file": ("malicious.zip", malicious_buf.getvalue(), "application/zip")}
    res = client.post("/api/v1/projects/import", files=files)
    assert res.status_code == 422
    assert "path traversal" in res.json()["error"]["message"].lower()
