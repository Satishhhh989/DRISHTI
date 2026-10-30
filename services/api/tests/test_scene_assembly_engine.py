"""DRISHTI - Scene Assembly Engine Tests
Tests deterministic multi-person placement, horizontal preservation,
depth ordering, occlusion handling, partial reconstruction resilience,
and transform validation.
"""

import pytest
from app.pipeline.assembly import (
    AssemblyConfig,
    SceneAssemblyEngine,
    compute_2d_iou,
    validate_transform_values,
)
from app.schemas.assets import AssetDimensions, AssetMetadata
from app.schemas.common import BoundingBox2D, Transform, Vector3D
from app.schemas.scene import (
    EntityDepthInfo,
    GeometryMetadata,
    HumanReconstructionData,
    MeshBounds,
    ModelProvenance,
    Scene,
    SceneEntity,
    SourceImageMetadata,
)


def create_dummy_scene(aspect_ratio: float = 1.333) -> Scene:
    width = int(aspect_ratio * 1000)
    height = 1000
    source_meta = SourceImageMetadata(
        asset_id="asset-test-photo",
        filename="test.jpg",
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        mime_type="image/jpeg",
        size_bytes=100000,
    )
    return Scene(
        id="scene-assembly-test",
        name="Test Multi-Person Scene",
        source_image=source_meta,
        entities=[],
    )


def test_horizontal_placement_ordering():
    """Verifies that Person A (left), Person B (center), and Person C (right) preserve horizontal ordering."""
    engine = SceneAssemblyEngine()
    scene = create_dummy_scene(aspect_ratio=1.5)

    # Person 1 on left (x: 0.1 to 0.3, center: 0.2)
    p_left = SceneEntity(
        id="person-001",
        source_image_id="asset-test-photo",
        name="Person Left",
        bounding_box=BoundingBox2D(x_min=0.1, y_min=0.2, x_max=0.3, y_max=0.8),
    )
    # Person 2 in center (x: 0.4 to 0.6, center: 0.5)
    p_center = SceneEntity(
        id="person-002",
        source_image_id="asset-test-photo",
        name="Person Center",
        bounding_box=BoundingBox2D(x_min=0.4, y_min=0.2, x_max=0.6, y_max=0.8),
    )
    # Person 3 on right (x: 0.7 to 0.9, center: 0.8)
    p_right = SceneEntity(
        id="person-003",
        source_image_id="asset-test-photo",
        name="Person Right",
        bounding_box=BoundingBox2D(x_min=0.7, y_min=0.2, x_max=0.9, y_max=0.8),
    )

    scene.entities = [p_left, p_center, p_right]
    result = engine.assemble(scene)

    assert result.success is True
    assert len(result.entities) == 3

    e_left = next(e for e in result.entities if e.id == "person-001")
    e_center = next(e for e in result.entities if e.id == "person-002")
    e_right = next(e for e in result.entities if e.id == "person-003")

    # X positions must be strictly monotonically increasing: X_left < X_center < X_right
    assert e_left.transform.position.x < e_center.transform.position.x
    assert e_center.transform.position.x < e_right.transform.position.x

    # Center person should be close to x = 0.0
    assert abs(e_center.transform.position.x) < 0.05
    # Left person should be negative X
    assert e_left.transform.position.x < 0.0
    # Right person should be positive X
    assert e_right.transform.position.x > 0.0


def test_depth_placement_with_relative_proximity():
    """Verifies that higher relative proximity puts an entity farther forward (+Z)."""
    engine = SceneAssemblyEngine(AssemblyConfig(depth_range_meters=2.0))
    scene = create_dummy_scene()

    # Person 1 is in foreground (proximity = 0.9)
    p_foreground = SceneEntity(
        id="person-fg",
        source_image_id="asset-test-photo",
        bounding_box=BoundingBox2D(x_min=0.2, y_min=0.2, x_max=0.4, y_max=0.8),
        depth=EntityDepthInfo(median_proximity=0.9, valid_pixel_coverage=1.0),
    )
    # Person 2 is in background (proximity = 0.2)
    p_background = SceneEntity(
        id="person-bg",
        source_image_id="asset-test-photo",
        bounding_box=BoundingBox2D(x_min=0.6, y_min=0.2, x_max=0.8, y_max=0.8),
        depth=EntityDepthInfo(median_proximity=0.2, valid_pixel_coverage=1.0),
    )

    scene.entities = [p_foreground, p_background]
    result = engine.assemble(scene)

    assert result.success is True
    e_fg = next(e for e in result.entities if e.id == "person-fg")
    e_bg = next(e for e in result.entities if e.id == "person-bg")

    # Foreground entity (+Z) must have greater Z than background entity
    assert e_fg.transform.position.z > e_bg.transform.position.z
    assert e_fg.transform.position.z > 0.0
    assert e_bg.transform.position.z < 0.0
    assert e_fg.placement.depth_source == "relative_median_proximity"
    assert e_fg.placement.is_metric is False


def test_missing_depth_fallback():
    """Verifies fallback when depth is unavailable for one entity."""
    engine = SceneAssemblyEngine()
    scene = create_dummy_scene()

    p_no_depth = SceneEntity(
        id="person-nodepth",
        source_image_id="asset-test-photo",
        bounding_box=BoundingBox2D(x_min=0.3, y_min=0.2, x_max=0.5, y_max=0.8),
        depth=None,
    )

    scene.entities = [p_no_depth]
    result = engine.assemble(scene)

    assert result.success is True
    e = result.entities[0]
    assert e.transform.position.z == 0.0
    assert e.placement.depth_source == "fallback_plane"
    assert "missing_depth" in e.placement.uncertainty_flags


def test_occlusion_ambiguity_warning():
    """Verifies that overlapping bounding boxes with very close relative depths generate occlusion warnings."""
    engine = SceneAssemblyEngine(AssemblyConfig(occlusion_iou_threshold=0.1, occlusion_depth_margin=0.05))
    scene = create_dummy_scene()

    # Overlapping boxes: x: [0.3, 0.6] and [0.4, 0.7]
    p1 = SceneEntity(
        id="person-001",
        source_image_id="asset-test-photo",
        bounding_box=BoundingBox2D(x_min=0.3, y_min=0.2, x_max=0.6, y_max=0.8),
        depth=EntityDepthInfo(median_proximity=0.55),
    )
    p2 = SceneEntity(
        id="person-002",
        source_image_id="asset-test-photo",
        bounding_box=BoundingBox2D(x_min=0.4, y_min=0.2, x_max=0.7, y_max=0.8),
        depth=EntityDepthInfo(median_proximity=0.57),  # |ΔP| = 0.02 < 0.05
    )

    scene.entities = [p1, p2]
    result = engine.assemble(scene)

    assert result.success is True
    assert len(result.report.occlusion_warnings) >= 1
    assert "single-view ambiguity" in result.report.occlusion_warnings[0]


def test_partial_reconstruction_resilience():
    """Verifies that an entity without a 3D mesh is preserved in the scene graph without crashing or vanishing."""
    engine = SceneAssemblyEngine()
    scene = create_dummy_scene()

    # Person 1 has a reconstructed mesh
    p1 = SceneEntity(
        id="person-reconstructed",
        source_image_id="asset-test-photo",
        bounding_box=BoundingBox2D(x_min=0.1, y_min=0.2, x_max=0.4, y_max=0.8),
        reconstruction=HumanReconstructionData(
            status="ready",
            mesh_asset_id="mesh-asset-123",
            mesh_url="/api/v1/assets/meshes/test.glb",
            geometry_metadata=GeometryMetadata(vertex_count=6890, face_count=13776),
        ),
    )
    # Person 2 failed reconstruction
    p2 = SceneEntity(
        id="person-failed",
        source_image_id="asset-test-photo",
        bounding_box=BoundingBox2D(x_min=0.6, y_min=0.2, x_max=0.9, y_max=0.8),
        reconstruction=HumanReconstructionData(
            status="failed",
            error_message="Severe occlusion of limbs",
        ),
    )

    scene.entities = [p1, p2]
    result = engine.assemble(scene)

    assert result.success is True
    assert result.report.status == "partial"
    assert result.report.total_entities == 2
    assert result.report.reconstructed_count == 1
    assert result.report.unreconstructed_count == 1

    # Both entities exist in SceneGraph
    assert "node-person-reconstructed" in result.scene_graph.nodes
    assert "node-person-failed" in result.scene_graph.nodes

    node_rec = result.scene_graph.nodes["node-person-reconstructed"]
    node_failed = result.scene_graph.nodes["node-person-failed"]

    assert node_rec.mesh_asset_id == "mesh-asset-123"
    assert node_failed.mesh_asset_id is None
    assert "mesh_not_reconstructed" in node_failed.provenance.uncertainty_flags


def test_transform_validation_utility():
    """Verifies that transform validator rejects NaNs, infinities, and non-positive scales."""
    assert validate_transform_values((0.0, 1.0, 2.0), (1.0, 1.0, 1.0)) is True
    assert validate_transform_values((float("nan"), 1.0, 2.0), (1.0, 1.0, 1.0)) is False
    assert validate_transform_values((0.0, float("inf"), 2.0), (1.0, 1.0, 1.0)) is False
    assert validate_transform_values((0.0, 1.0, 2.0), (0.0, 1.0, 1.0)) is False
    assert validate_transform_values((0.0, 1.0, 2.0), (-1.0, 1.0, 1.0)) is False
    assert validate_transform_values((150.0, 1.0, 2.0), (1.0, 1.0, 1.0)) is False


def test_scene_graph_hierarchy():
    """Verifies root contains environment and person node references."""
    engine = SceneAssemblyEngine()
    scene = create_dummy_scene()
    p1 = SceneEntity(
        id="p1",
        source_image_id="asset-test-photo",
        bounding_box=BoundingBox2D(x_min=0.3, y_min=0.3, x_max=0.7, y_max=0.7),
    )
    scene.entities = [p1]
    result = engine.assemble(scene)

    root = result.scene_graph.nodes["scene-root"]
    assert "camera-main" in root.children
    assert "light-ambient" in root.children
    assert "light-directional" in root.children
    assert "plane-reference" in root.children
    assert "plane-ground" in root.children
    assert "node-p1" in root.children

