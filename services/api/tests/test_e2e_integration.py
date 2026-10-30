"""DRISHTI - End-to-End System Integration Tests
Phase 10: Integration, Testing, Security Hardening, Final Verification & Release
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
from app.pipeline.coordinator import PipelineCoordinator
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
    PlacementProvenance,
    SkeletonData,
    SkeletonJoint,
    RigData,
    AnimationClipData,
    JointTransformTrack,
)

client = TestClient(app)


def test_full_lifecycle_integration(tmp_path):
    """Executes the complete DRISHTI workflow through all connected subsystems."""

    # 1. Image Asset Creation
    photo_asset = AssetMetadata(
        asset_id="asset-photo-e2e-final",
        category="original_photo",
        filename="friends_e2e.jpg",
        relative_path="photos/friends_e2e.jpg",
        mime_type="image/jpeg",
        size_bytes=8192,
        checksum_sha256="d" * 64,
        dimensions=AssetDimensions(width=1600, height=1200),
    )

    # 2. Scene Initialization
    scene = scene_manager.create_scene_from_photo(photo_asset, name="E2E Release Scene")
    assert scene.id is not None
    assert scene.representations.photo.status == "ready"

    # 3. Add genuine 3D mesh assets and multi-person entities
    box1 = trimesh.creation.box(extents=[0.5, 1.7, 0.3])
    glb_bytes1 = export_mesh_glb(box1.vertices, box1.faces, name="person_0")
    mesh1 = asset_manager.register_mesh(scene_id=scene.id, content=glb_bytes1, filename="p0_mesh.glb")

    box2 = trimesh.creation.box(extents=[0.6, 1.8, 0.35])
    glb_bytes2 = export_mesh_glb(box2.vertices, box2.faces, name="person_1")
    mesh2 = asset_manager.register_mesh(scene_id=scene.id, content=glb_bytes2, filename="p1_mesh.glb")

    # Person 0 (Reconstructed & Rigged)
    e0 = SceneEntity(
        id="person-0",
        source_image_id=photo_asset.asset_id,
        name="Alice",
        reconstruction_status="ready",
        transform=Transform(
            position=Vector3D(x=-0.6, y=0.0, z=-1.4),
            rotation=EulerRotation(x=0.0, y=0.15, z=0.0),
            scale=Vector3D(x=1.0, y=1.0, z=1.0),
        ),
        placement=PlacementProvenance(
            image_space_coords=[0.3, 0.5],
            relative_proximity=0.85,
            orientation_correction=[0.0, 0.15, 0.0],
        ),
        reconstruction=HumanReconstructionData(
            status="ready",
            mesh_asset_id=mesh1.asset_id,
            mesh_url=f"/api/v1/assets/{mesh1.asset_id}/file",
            geometry_metadata=GeometryMetadata(
                vertex_count=len(box1.vertices),
                face_count=len(box1.faces),
                bounds=MeshBounds(min=[-0.25, -0.85, -0.15], max=[0.25, 0.85, 0.15]),
                dimensions=[0.5, 1.7, 0.3],
            ),
        ),
        skeleton=SkeletonData(
            id="skel-p0",
            entity_id="person-0",
            joints=[
                SkeletonJoint(id="pelvis", name="pelvis", index=0, position=[0.0, 0.0, 0.0], is_observed=True),
                SkeletonJoint(id="spine1", name="spine1", index=1, parent_id="pelvis", position=[0.0, 0.2, 0.0], is_observed=True),
            ],
        ),
        rig=RigData(
            id="rig-p0",
            entity_id="person-0",
            skeleton_id="skel-p0",
            mesh_asset_id=mesh1.asset_id,
            status="rigged",
            skinning_method="smpl_lbs_native",
        ),
        animation_clips=[
            AnimationClipData(
                id="clip-idle-p0",
                name="idle_breathing",
                target_entity_id="person-0",
                target_skeleton_id="skel-p0",
                duration_seconds=3.0,
                tracks=[
                    JointTransformTrack(
                        joint_id="pelvis",
                        timestamps=[0.0, 1.5, 3.0],
                        rotations=[[0.0, 0.0, 0.0, 1.0], [0.0, 0.05, 0.0, 0.998], [0.0, 0.0, 0.0, 1.0]],
                    )
                ],
            )
        ],
    )

    # Person 1 (Reconstructed)
    e1 = SceneEntity(
        id="person-1",
        source_image_id=photo_asset.asset_id,
        name="Bob",
        reconstruction_status="ready",
        transform=Transform(
            position=Vector3D(x=0.5, y=0.0, z=-1.8),
            rotation=EulerRotation(x=0.0, y=-0.1, z=0.0),
            scale=Vector3D(x=1.0, y=1.0, z=1.0),
        ),
        placement=PlacementProvenance(
            image_space_coords=[0.7, 0.5],
            relative_proximity=0.6,
            orientation_correction=[0.0, -0.1, 0.0],
        ),
        reconstruction=HumanReconstructionData(
            status="ready",
            mesh_asset_id=mesh2.asset_id,
            mesh_url=f"/api/v1/assets/{mesh2.asset_id}/file",
            geometry_metadata=GeometryMetadata(
                vertex_count=len(box2.vertices),
                face_count=len(box2.faces),
                bounds=MeshBounds(min=[-0.3, -0.9, -0.175], max=[0.3, 0.9, 0.175]),
                dimensions=[0.6, 1.8, 0.35],
            ),
        ),
    )

    scene.entities = [e0, e1]
    scene.representations.wireframe.status = "ready"
    scene.representations.solid3d.status = "ready"
    scene.representations.articulated.status = "ready"
    scene.representations.combined.status = "ready"
    scene_manager.update_scene(scene)

    # 4. Project Initialization & Persistence
    project = project_manager.create_project_from_scene(scene, name="Final Release Project")
    assert project.id.startswith("proj-")
    assert project.scene_id == scene.id
    assert len(project.asset_manifest) >= 2  # Photo + 2 meshes

    # 5. Interactive Transform Modification
    patch_res = client.patch(
        f"/api/v1/scenes/{scene.id}/entities/person-0/transform",
        json={"position": {"x": -0.8, "y": 0.0, "z": -1.2}},
    )
    assert patch_res.status_code == 200
    refreshed_scene = patch_res.json()
    e0_after = next(e for e in refreshed_scene["entities"] if e["id"] == "person-0")
    assert e0_after["transform"]["position"]["x"] == -0.8

    # 6. Save Project
    save_res = client.put(
        f"/api/v1/projects/{project.id}",
        json={"name": "Final Release Project (Saved)"},
    )
    assert save_res.status_code == 200
    assert save_res.json()["name"] == "Final Release Project (Saved)"

    # 7. Reload from Disk & Verify Identity Preservation
    pm_clean = project_manager
    reloaded_project = pm_clean.get_project(project.id)
    assert reloaded_project.name == "Final Release Project (Saved)"
    reloaded_scene = scene_manager.get_scene(scene.id)
    assert len(reloaded_scene.entities) == 2
    assert reloaded_scene.entities[0].id == "person-0"
    assert reloaded_scene.entities[1].id == "person-1"

    # 8. Real Asset Exports
    # a. Single Entity Mesh GLB
    mesh_res = client.get(f"/api/v1/export/mesh/{scene.id}/person-0")
    assert mesh_res.status_code == 200
    assert mesh_res.headers["content-type"] == "model/gltf-binary"
    m_loaded = trimesh.load(trimesh.util.wrap_as_stream(mesh_res.content), file_type="glb")
    assert m_loaded is not None

    # b. Assembled Multi-Person Scene GLB
    scene_glb_res = client.get(f"/api/v1/export/scene/{scene.id}")
    assert scene_glb_res.status_code == 200
    s_loaded = trimesh.load(trimesh.util.wrap_as_stream(scene_glb_res.content), file_type="glb")
    assert isinstance(s_loaded, trimesh.Scene)
    assert len(s_loaded.geometry) == 2

    # c. Scene Metadata JSON
    meta_res = client.get(f"/api/v1/export/metadata/{scene.id}")
    assert meta_res.status_code == 200
    meta_json = meta_res.json()
    assert meta_json["schema_version"] == "1.0.0"
    assert len(meta_json["entities"]) == 2

    # d. Portable Project Archive ZIP
    zip_res = client.get(f"/api/v1/export/project/{project.id}/archive")
    assert zip_res.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(zip_res.content))
    assert "project.json" in zf.namelist()
    assert "scene.json" in zf.namelist()
    assert "manifest.json" in zf.namelist()

    # 9. Reimport Archive Safely
    files = {"file": ("portable.zip", zip_res.content, "application/zip")}
    data = {"new_name": "Reimported Verified Project"}
    imp_res = client.post("/api/v1/projects/import", files=files, data=data)
    assert imp_res.status_code == 200
    imp_data = imp_res.json()
    assert imp_data["name"] == "Reimported Verified Project"
    assert imp_data["id"].startswith("proj-imp-")

