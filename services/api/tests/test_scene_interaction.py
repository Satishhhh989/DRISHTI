"""DRISHTI - Scene Interaction & Transform Tests
Phase 09: Interaction, Persistence & Export
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.scene.manager import scene_manager
from app.schemas.assets import AssetDimensions, AssetMetadata
from app.schemas.common import Transform, Vector3D, EulerRotation
from app.schemas.scene import SceneEntity, PlacementProvenance


@pytest.fixture
def test_scene():
    asset = AssetMetadata(
        asset_id="asset-photo-interaction",
        category="original_photo",
        filename="interaction_test.jpg",
        relative_path="photos/interaction_test.jpg",
        mime_type="image/jpeg",
        size_bytes=4096,
        checksum_sha256="b" * 64,
        dimensions=AssetDimensions(width=1000, height=800),
    )
    scene = scene_manager.create_scene_from_photo(asset, name="Interaction Scene")

    # Add two entities
    e1 = SceneEntity(
        id="person-0",
        source_image_id=asset.asset_id,
        name="Subject 0",
        transform=Transform(
            position=Vector3D(x=-0.5, y=0.0, z=-1.2),
            rotation=EulerRotation(x=0.0, y=0.1, z=0.0),
            scale=Vector3D(x=1.0, y=1.0, z=1.0),
        ),
        placement=PlacementProvenance(
            image_space_coords=[0.3, 0.5],
            relative_proximity=0.8,
            orientation_correction=[0.0, 0.1, 0.0],
        ),
    )
    e2 = SceneEntity(
        id="person-1",
        source_image_id=asset.asset_id,
        name="Subject 1",
        transform=Transform(
            position=Vector3D(x=0.6, y=0.0, z=-1.8),
            rotation=EulerRotation(x=0.0, y=-0.2, z=0.0),
            scale=Vector3D(x=1.0, y=1.0, z=1.0),
        ),
        placement=PlacementProvenance(
            image_space_coords=[0.7, 0.5],
            relative_proximity=0.5,
            orientation_correction=[0.0, -0.2, 0.0],
        ),
    )
    scene.entities = [e1, e2]
    scene_manager.update_scene(scene)
    return scene


client = TestClient(app)


def test_update_entity_transform_valid(test_scene):
    payload = {
        "position": {"x": 1.5, "y": 0.2, "z": -2.0},
        "rotation": {"x": 0.0, "y": 0.5, "z": 0.0},
        "scale": {"x": 1.2, "y": 1.2, "z": 1.2},
    }
    res = client.patch(
        f"/api/v1/scenes/{test_scene.id}/entities/person-0/transform",
        json=payload,
    )
    assert res.status_code == 200
    data = res.json()
    e0 = next(e for e in data["entities"] if e["id"] == "person-0")
    assert e0["transform"]["position"]["x"] == 1.5
    assert e0["transform"]["position"]["y"] == 0.2
    assert e0["transform"]["position"]["z"] == -2.0
    assert e0["transform"]["scale"]["x"] == 1.2


def test_reject_out_of_bounds_transform(test_scene):
    # Position > 50 meters
    res = client.patch(
        f"/api/v1/scenes/{test_scene.id}/entities/person-0/transform",
        json={"position": {"x": 999.0, "y": 0.0, "z": 0.0}},
    )
    assert res.status_code == 422
    assert "exceeds permissible scene bounds" in res.json()["error"]["message"]


def test_reject_invalid_scale(test_scene):
    # Scale <= 0
    res = client.patch(
        f"/api/v1/scenes/{test_scene.id}/entities/person-0/transform",
        json={"scale": {"x": -1.0, "y": 1.0, "z": 1.0}},
    )
    assert res.status_code == 422

    # Scale > 10.0
    res2 = client.patch(
        f"/api/v1/scenes/{test_scene.id}/entities/person-0/transform",
        json={"scale": {"x": 25.0, "y": 1.0, "z": 1.0}},
    )
    assert res2.status_code == 422


def test_reset_entity_transform(test_scene):
    # First modify transform
    client.patch(
        f"/api/v1/scenes/{test_scene.id}/entities/person-0/transform",
        json={"position": {"x": 10.0, "y": 5.0, "z": -10.0}},
    )

    # Call reset
    res = client.post(
        f"/api/v1/scenes/{test_scene.id}/entities/person-0/reset-transform"
    )
    assert res.status_code == 200
    data = res.json()
    e0 = next(e for e in data["entities"] if e["id"] == "person-0")

    # Should be back to placement coordinates: (0.3 - 0.5) * 2.0 = -0.4
    assert round(e0["transform"]["position"]["x"], 1) == -0.4
    assert e0["transform"]["position"]["y"] == 0.0
    assert e0["transform"]["scale"]["x"] == 1.0


def test_delete_scene_entity(test_scene):
    res = client.delete(f"/api/v1/scenes/{test_scene.id}/entities/person-1")
    assert res.status_code == 200
    data = res.json()
    assert len(data["entities"]) == 1
    assert data["entities"][0]["id"] == "person-0"

    # Deleting non-existent returns 404
    res_404 = client.delete(f"/api/v1/scenes/{test_scene.id}/entities/person-99")
    assert res_404.status_code == 404


def test_keep_only_scene_entities(test_scene):
    res = client.post(
        f"/api/v1/scenes/{test_scene.id}/entities/keep-only",
        json={"keep_entity_ids": ["person-1"]},
    )
    assert res.status_code == 200
    data = res.json()
    assert len(data["entities"]) == 1
    assert data["entities"][0]["id"] == "person-1"

