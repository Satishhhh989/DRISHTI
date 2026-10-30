"""DRISHTI - Scene Persistence Tests
Tests local disk serialization, recovery after restart, schema versioning,
and transform preservation.
"""

import json
from pathlib import Path
import tempfile
import pytest

from app.scene.manager import SceneManager
from app.schemas.assets import AssetDimensions, AssetMetadata
from app.schemas.common import BoundingBox2D, Transform, Vector3D
from app.schemas.scene import (
    GeometryMetadata,
    HumanReconstructionData,
    PlacementProvenance,
    SceneEntity,
)


def test_scene_save_and_reload(tmp_path: Path):
    """Verifies that an assembled scene can be saved to disk, reloaded, and preserves all entity transforms and mesh references."""
    manager = SceneManager(storage_dir=tmp_path)

    asset = AssetMetadata(
        asset_id="asset-photo-100",
        category="original_photo",
        filename="group_photo.jpg",
        relative_path="photos/group_photo.jpg",
        mime_type="image/jpeg",
        dimensions=AssetDimensions(width=1600, height=1200),
        size_bytes=250000,
        checksum_sha256="a" * 64,
    )

    scene = manager.create_scene_from_photo(asset, name="Group Scene")

    # Add person entity with transform, placement, and reconstruction
    person = SceneEntity(
        id="person-001",
        source_image_id=asset.asset_id,
        name="Person 1",
        bounding_box=BoundingBox2D(x_min=0.2, y_min=0.2, x_max=0.5, y_max=0.8),
        transform=Transform(
            position=Vector3D(x=-0.5, y=1.0, z=0.35),
            scale=Vector3D(x=1.05, y=1.05, z=1.05),
        ),
        geometry_ref="mesh-person-001",
        reconstruction=HumanReconstructionData(
            status="ready",
            mesh_asset_id="mesh-person-001",
            mesh_url="/api/v1/assets/meshes/person_001.glb",
            geometry_metadata=GeometryMetadata(vertex_count=6890, face_count=13776),
        ),
        placement=PlacementProvenance(
            method="depth_projected_composition",
            image_space_coords=[0.35, 0.5],
            depth_source="relative_median_proximity",
            relative_proximity=0.72,
            model_normalization_applied="canonical_humanoid_scale",
            uncertainty_flags=[],
            is_metric=False,
        ),
    )
    scene.entities.append(person)
    manager.update_scene(scene)

    # Verify JSON file was written to disk
    scene_file = tmp_path / f"{scene.id}.json"
    assert scene_file.is_file()

    # Simulate backend restart with new SceneManager instance
    new_manager = SceneManager(storage_dir=tmp_path)
    loaded_scene = new_manager.get_scene(scene.id)

    assert loaded_scene.id == scene.id
    assert loaded_scene.name == "Group Scene"
    assert len(loaded_scene.entities) == 1

    loaded_person = loaded_scene.entities[0]
    assert loaded_person.id == "person-001"
    assert loaded_person.transform.position.x == -0.5
    assert loaded_person.transform.position.y == 1.0
    assert loaded_person.transform.position.z == 0.35
    assert loaded_person.transform.scale.x == 1.05
    assert loaded_person.geometry_ref == "mesh-person-001"
    assert loaded_person.reconstruction.geometry_metadata.vertex_count == 6890
    assert loaded_person.placement.relative_proximity == 0.72


def test_scene_list_discovers_persisted_files(tmp_path: Path):
    """Verifies list_scenes discovers files persisted from prior sessions."""
    manager1 = SceneManager(storage_dir=tmp_path)
    asset = AssetMetadata(
        asset_id="asset-1",
        category="original_photo",
        filename="p1.jpg",
        relative_path="photos/p1.jpg",
        mime_type="image/jpeg",
        dimensions=AssetDimensions(width=800, height=600),
        size_bytes=1000,
        checksum_sha256="b" * 64,
    )
    s1 = manager1.create_scene_from_photo(asset)

    # Another session
    manager2 = SceneManager(storage_dir=tmp_path)
    scenes = manager2.list_scenes()
    assert len(scenes) == 1
    assert scenes[0].id == s1.id
