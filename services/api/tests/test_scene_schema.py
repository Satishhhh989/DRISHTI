import pytest
from pydantic import ValidationError
from app.schemas.common import Transform, Vector3D, EulerRotation, BoundingBox2D
from app.schemas.scene import (
    Scene,
    SceneEntity,
    SourceImageMetadata,
    SceneRepresentations,
    RepresentationAsset,
    SCHEMA_VERSION,
)


def test_valid_scene_model():
    source_meta = SourceImageMetadata(
        asset_id="asset-photo-01",
        filename="group.jpg",
        width=1920,
        height=1080,
        aspect_ratio=1.777778,
        mime_type="image/jpeg",
        size_bytes=2048576,
    )

    reps = SceneRepresentations(
        photo=RepresentationAsset(status="ready", asset_id="asset-photo-01", uri="/api/v1/assets/asset-photo-01/file"),
        spatial_photo=RepresentationAsset(status="unavailable"),
        wireframe=RepresentationAsset(status="unavailable"),
        solid3d=RepresentationAsset(status="unavailable"),
    )

    scene = Scene(
        id="scene-001",
        name="Conference Photo",
        source_image=source_meta,
        representations=reps,
    )

    assert scene.id == "scene-001"
    assert scene.schema_version == SCHEMA_VERSION
    assert scene.coordinate_system.handedness == "right-handed"
    assert scene.representations.photo.status == "ready"
    assert scene.representations.wireframe.status == "unavailable"


def test_unsupported_schema_version():
    source_meta = SourceImageMetadata(
        asset_id="asset-01",
        filename="test.png",
        width=800,
        height=600,
        aspect_ratio=1.3333,
        mime_type="image/png",
        size_bytes=1000,
    )
    with pytest.raises(ValidationError) as exc_info:
        Scene(
            id="scene-invalid-ver",
            schema_version="9.9.9",
            name="Invalid Ver",
            source_image=source_meta,
        )
    assert "Unsupported schema version" in str(exc_info.value)


def test_transform_scale_validation():
    # Negative scale must fail
    with pytest.raises(ValidationError) as exc_info:
        Transform(
            position=Vector3D(x=0, y=1, z=0),
            rotation=EulerRotation(x=0, y=0, z=0),
            scale=Vector3D(x=1, y=-0.5, z=1),
        )
    assert "Transform scale components must be strictly positive" in str(exc_info.value)


def test_bounding_box_validation():
    # x_min >= x_max must fail
    with pytest.raises(ValidationError) as exc_info:
        BoundingBox2D(x_min=0.8, y_min=0.1, x_max=0.3, y_max=0.9)
    assert "must be strictly less than x_max" in str(exc_info.value)

