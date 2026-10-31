"""DRISHTI - Object 3D Reconstruction Integration & Validation Tests
"""

from pathlib import Path
import numpy as np
import pytest
import trimesh

from app.pipeline.object_reconstructor import (
    CATEGORY_DIMENSION_PRIORS,
    Object3DReconstructor,
    object_reconstructor,
)
from app.pipeline.detector import YoloPersonDetector
from app.pipeline.segmentor import YoloPersonSegmentor
from app.pipeline.assembly import SceneAssemblyEngine
from app.schemas.common import BoundingBox2D, Transform, Vector3D
from app.schemas.scene import (
    GeometryMetadata,
    HumanReconstructionData,
    MeshBounds,
    Scene,
    SceneEntity,
)


def test_object_reconstructor_generates_valid_glb():
    """Validates that Object3DReconstructor produces non-empty, finite, textured GLB geometry."""
    img = np.full((120, 120, 3), 150, dtype=np.uint8)
    # Draw a blue object
    img[30:90, 30:90] = [30, 80, 220]

    mask = np.zeros((120, 120), dtype=np.uint8)
    mask[30:90, 30:90] = 255

    result = object_reconstructor.reconstruct_object(
        image_rgb=img,
        mask_uint8=mask,
        normalized_depth=None,
        pixel_box=(30.0, 30.0, 90.0, 90.0),
        category_name="chair",
        entity_id="test-obj-01",
    )

    assert result.vertex_count > 0
    assert result.face_count > 0
    assert len(result.glb_bytes) > 1000
    assert result.category == "chair"
    assert len(result.dimensions) == 3
    assert all(d > 0 for d in result.dimensions)

    # Load with trimesh to inspect geometry
    loaded = trimesh.load(trimesh.util.wrap_as_stream(result.glb_bytes), file_type="glb")
    # GLB contains Scene or Trimesh
    mesh = list(loaded.geometry.values())[0] if isinstance(loaded, trimesh.Scene) else loaded
    assert len(mesh.vertices) == result.vertex_count
    assert np.all(np.isfinite(mesh.vertices))


def test_category_dimension_priors_coverage():
    """Ensures major object categories have metric spatial priors."""
    for cat in ["bus", "car", "chair", "dining table", "laptop", "bottle"]:
        assert cat in CATEGORY_DIMENSION_PRIORS
        w, h, d = CATEGORY_DIMENSION_PRIORS[cat]
        assert w > 0 and h > 0 and d > 0


def test_object_scene_assembly_node_type():
    """Validates that SceneAssemblyEngine produces node_type='object' for object entities."""
    engine = SceneAssemblyEngine()

    obj_entity = SceneEntity(
        id="object-chair-01",
        entity_type="object",
        object_class="chair",
        source_image_id="photo-001",
        name="Chair 1",
        bounding_box=BoundingBox2D(x_min=0.2, y_min=0.4, x_max=0.5, y_max=0.8),
        transform=Transform(),
        reconstruction=HumanReconstructionData(
            status="ready",
            mesh_asset_id="mesh-obj-01",
            mesh_url="/api/v1/assets/meshes/mesh-obj-01.glb",
            geometry_metadata=GeometryMetadata(
                vertex_count=1200,
                face_count=2400,
                bounds=MeshBounds(min=[-0.3, -0.4, -0.3], max=[0.3, 0.4, 0.3]),
                dimensions=[0.6, 0.8, 0.6],
            ),
        ),
    )

    from app.schemas.scene import SourceImageMetadata

    scene = Scene(
        id="scene-object-test",
        name="Object Test Scene",
        source_image=SourceImageMetadata(
            asset_id="photo-001",
            filename="test.jpg",
            width=1200,
            height=900,
            aspect_ratio=1.333,
            mime_type="image/jpeg",
            size_bytes=1000,
        ),
        entities=[obj_entity],
    )

    assembly_result = engine.assemble(scene)
    assert "node-object-chair-01" in assembly_result.scene_graph.nodes
    node = assembly_result.scene_graph.nodes["node-object-chair-01"]
    assert node.node_type == "object"
    assert node.entity_id == "object-chair-01"
    assert node.mesh_asset_id == "mesh-obj-01"
