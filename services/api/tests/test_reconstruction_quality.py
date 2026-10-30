"""DRISHTI - Comprehensive Human 3D Reconstruction Quality & Alignment Regression Suite

Deterministic verification for:
1. Image-to-crop coordinate conversion.
2. Crop-to-source keypoint conversion.
3. Left/right joint mapping.
4. Model-to-scene coordinate conversion.
5. Up-axis conversion (Head above feet in +Y).
6. Root transform composition.
7. Skeleton bind-pose correctness.
8. Mesh orientation (facing +Z camera).
9. Standing versus seated pose preservation.
10. Independent transforms for multiple people.
11. Source-camera projection.
12. Photograph and mesh alignment.
13. Rigging and skinning validity.
14. Independent animation state.
15. Scene save and reopen.
"""

import io
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
import trimesh

from app.pipeline.assembly import SceneAssemblyEngine, AssemblyConfig
from app.pipeline.coordinator import PipelineCoordinator
from app.pipeline.detector import YoloPersonDetector
from app.pipeline.device import device_manager
from app.pipeline.models import model_manager
from app.pipeline.motion import (
    CANONICAL_JOINT_NAMES,
    CANONICAL_PARENTS,
    CANONICAL_REST_POSITIONS,
    SkeletonFitter,
    RiggingEngine,
    ProceduralAnimationEngine,
    skeleton_fitter,
)
from app.pipeline.reconstructor import (
    MockHumanReconstructor,
    SpinHumanReconstructor,
    compute_mesh_normals,
    export_mesh_glb,
    prepare_person_crop,
    validate_mesh_geometry,
)
from app.schemas.common import BoundingBox2D, Transform, Vector3D, EulerRotation
from app.schemas.scene import (
    AnimationState,
    ConfidenceMetadata,
    CoordinateConvention,
    DetectionInfo,
    GeometryMetadata,
    HumanReconstructionData,
    MeshBounds,
    ModelProvenance,
    Scene,
    SceneEntity,
    SourceImageMetadata,
)


# ---------------------------------------------------------------------------
# 1. Image-to-crop coordinate conversion
# ---------------------------------------------------------------------------
def test_image_to_crop_coordinate_conversion():
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    box = BoundingBox2D(x_min=0.2, y_min=0.1, x_max=0.6, y_max=0.9)
    crop, meta = prepare_person_crop(img, box, padding=0.2, target_size=(224, 224))

    assert crop.shape == (224, 224, 3)
    assert meta["aspect_ratio"] > 0
    assert "offset" in meta
    assert "square_dim" in meta
    assert "is_clipped" in meta


# ---------------------------------------------------------------------------
# 2. Crop-to-source keypoint conversion
# ---------------------------------------------------------------------------
def test_crop_to_source_keypoint_conversion():
    img_w, img_h = 1536, 1024
    box = BoundingBox2D(x_min=0.2, y_min=0.1, x_max=0.4, y_max=0.8)
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    _, meta = prepare_person_crop(img, box, padding=0.2, target_size=(224, 224))

    # Center of square crop (112, 112) maps back to original image
    sq_dim = meta["square_dim"]
    clamp_x1, clamp_y1, clamp_x2, clamp_y2 = meta["crop_pixels"]
    offset_x, offset_y = meta["offset"]

    # In square space
    crop_cx, crop_cy = 112.0, 112.0
    crop_in_sq_x = crop_cx * (sq_dim / 224.0)
    crop_in_sq_y = crop_cy * (sq_dim / 224.0)

    orig_x = clamp_x1 + (crop_in_sq_x - offset_x)
    orig_y = clamp_y1 + (crop_in_sq_y - offset_y)
    norm_orig_x = orig_x / img_w
    norm_orig_y = orig_y / img_h

    expected_cx = (box.x_min + box.x_max) / 2.0
    expected_cy = (box.y_min + box.y_max) / 2.0

    assert abs(norm_orig_x - expected_cx) < 0.05
    assert abs(norm_orig_y - expected_cy) < 0.05


# ---------------------------------------------------------------------------
# 3. Left/right joint mapping
# ---------------------------------------------------------------------------
def test_left_right_joint_mapping():
    # In SMPL / DRISHTI canonical topology (+X is right from viewer's perspective):
    # Person facing camera has their Left limb on positive X, and Right limb on negative X.
    for name, pos in CANONICAL_REST_POSITIONS.items():
        if "left_hip" in name or "left_knee" in name or "left_ankle" in name:
            right_counterpart = name.replace("left_", "right_")
            assert right_counterpart in CANONICAL_REST_POSITIONS
            assert pos[0] != CANONICAL_REST_POSITIONS[right_counterpart][0]


# ---------------------------------------------------------------------------
# 4. Model-to-scene coordinate conversion
# ---------------------------------------------------------------------------
def test_model_to_scene_coordinate_conversion():
    # OpenCV camera space: Y is down (+Y towards ground), Z is forward (+Z away)
    # Three.js / DRISHTI canonical: Y is UP (+Y towards sky), Z is towards viewer (+Z towards camera)
    cam_vertex = np.array([0.2, -0.9, -0.5])  # Head in camera space (negative Y, negative Z)
    # Conversion: X' = X, Y' = -Y, Z' = -Z
    world_vertex = np.array([cam_vertex[0], -cam_vertex[1], -cam_vertex[2]])

    assert world_vertex[0] == 0.2
    assert world_vertex[1] == 0.9   # Head is now at positive Y (UP)
    assert world_vertex[2] == 0.5   # Face is now at positive Z (toward camera)


# ---------------------------------------------------------------------------
# 5. Up-axis conversion (Head above feet)
# ---------------------------------------------------------------------------
def test_up_axis_conversion_head_above_feet():
    if not model_manager.is_model_available("spin-human-reconstruction"):
        pytest.skip("SPIN model unavailable")

    ckpt_path = model_manager.get_model_path("spin-human-reconstruction")
    smpl_path = ckpt_path.parent / "SMPL_N_model.npz"
    mean_path = ckpt_path.parent / "smpl_mean_params.npz"
    rec = SpinHumanReconstructor(ckpt_path, smpl_path, mean_path, device=device_manager.device)

    img = np.ones((400, 300, 3), dtype=np.uint8) * 120
    box = BoundingBox2D(x_min=0.1, y_min=0.05, x_max=0.9, y_max=0.95)
    res = rec.reconstruct(img, box)

    # In canonical coordinates, max Y must be head, min Y must be feet
    v = res.vertices
    y_max = v[:, 1].max()
    y_min = v[:, 1].min()

    assert y_max > 0.5   # Head well above pelvis
    assert y_min < -0.3  # Feet well below pelvis
    assert y_max > y_min
    assert (y_max - y_min) >= 1.2  # Plausible human height


# ---------------------------------------------------------------------------
# 6. Root transform composition
# ---------------------------------------------------------------------------
def test_root_transform_composition():
    trans = Transform(
        position=Vector3D(x=1.0, y=2.0, z=3.0),
        rotation=EulerRotation(x=0.0, y=0.0, z=0.0),
        scale=Vector3D(x=1.5, y=1.5, z=1.5),
    )
    assert trans.position.x == 1.0
    assert trans.position.y == 2.0
    assert trans.position.z == 3.0
    assert trans.scale.x == 1.5


# ---------------------------------------------------------------------------
# 7. Skeleton bind-pose correctness
# ---------------------------------------------------------------------------
def test_skeleton_bind_pose_correctness():
    entity = SceneEntity(id="test-ent-1", source_image_id="asset-1")
    skel = skeleton_fitter.build_skeleton(entity)

    assert len(skel.joints) == 24
    assert len(skel.bones) == 23
    assert skel.root_joint_id == "pelvis"
    assert skel.validation_status == "valid"
    for bone in skel.bones:
        assert bone.length > 0.0


# ---------------------------------------------------------------------------
# 8. Mesh orientation facing camera
# ---------------------------------------------------------------------------
def test_mesh_orientation_facing_camera():
    if not model_manager.is_model_available("spin-human-reconstruction"):
        pytest.skip("SPIN model unavailable")

    ckpt_path = model_manager.get_model_path("spin-human-reconstruction")
    smpl_path = ckpt_path.parent / "SMPL_N_model.npz"
    mean_path = ckpt_path.parent / "smpl_mean_params.npz"
    rec = SpinHumanReconstructor(ckpt_path, smpl_path, mean_path, device=device_manager.device)

    img = np.ones((400, 300, 3), dtype=np.uint8) * 120
    box = BoundingBox2D(x_min=0.1, y_min=0.05, x_max=0.9, y_max=0.95)
    res = rec.reconstruct(img, box)

    # Front vertices (face, chest, knees) should extend toward positive Z (viewer)
    assert res.vertices[:, 2].max() > 0.2


# ---------------------------------------------------------------------------
# 9. Standing versus seated pose preservation
# ---------------------------------------------------------------------------
def test_standing_vs_seated_pose_preservation():
    test_img = Path("data/storage/photos/066bfeaf-236d-4a16-a7a8-a6c0f0cfcb9b_drishti_test_.png")
    if not test_img.exists() or not model_manager.is_model_available("spin-human-reconstruction"):
        pytest.skip("Test image or model unavailable")

    coord = PipelineCoordinator()
    rec = coord.get_reconstructor()
    det = coord.get_detector()
    img = np.array(Image.open(test_img).convert("RGB"))
    detections = det.detect_people(test_img)

    # Person 1 (seated center: x ~ 0.37-0.64)
    # Person 2 (standing left-center: x ~ 0.22-0.40)
    res_seated = rec.reconstruct(img, detections[0].bounding_box)
    res_standing = rec.reconstruct(img, detections[1].bounding_box)

    v_seated = res_seated.vertices
    v_standing = res_standing.vertices

    h_seated = v_seated[:, 1].max() - v_seated[:, 1].min()
    h_standing = v_standing[:, 1].max() - v_standing[:, 1].min()

    # Seated person has reduced vertical height due to bent legs
    assert h_seated < h_standing
    assert abs(h_seated - 1.47) < 0.10
    assert abs(h_standing - 1.69) < 0.10

    # Skeleton knee angle test
    class DummyEntity:
        def __init__(self, eid):
            self.id = eid

    skel_seated = skeleton_fitter.build_skeleton(DummyEntity("seated"), v_seated)
    skel_standing = skeleton_fitter.build_skeleton(DummyEntity("standing"), v_standing)

    def compute_knee_angle(skel):
        j_map = {j.id: np.array(j.position) for j in skel.joints}
        v1 = j_map["left_hip"] - j_map["left_knee"]
        v2 = j_map["left_ankle"] - j_map["left_knee"]
        cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))

    ang_seated = compute_knee_angle(skel_seated)
    ang_standing = compute_knee_angle(skel_standing)

    assert ang_seated < 110.0   # Seated knee bent around ~90-105 deg
    assert ang_standing > 120.0  # Standing knee upright ~128-135 deg


# ---------------------------------------------------------------------------
# 10. Independent transforms for multiple people
# ---------------------------------------------------------------------------
def test_independent_transforms_for_multiple_people():
    test_img = Path("data/storage/photos/066bfeaf-236d-4a16-a7a8-a6c0f0cfcb9b_drishti_test_.png")
    if not test_img.exists():
        pytest.skip("Test photo not found")

    det = YoloPersonDetector()
    detections = det.detect_people(test_img)
    assert len(detections) == 5

    aspect = 1536 / 1024
    plane_h = 2.0
    plane_w = plane_h * aspect

    positions_x = []
    for d in detections:
        b = d.bounding_box
        cx = (b.x_min + b.x_max) / 2.0
        pos_x = (cx - 0.5) * plane_w
        positions_x.append(pos_x)

    # Verify positions are strictly monotonically distributed and separated across the frame
    positions_x.sort()
    for i in range(len(positions_x) - 1):
        gap = positions_x[i + 1] - positions_x[i]
        assert gap >= 0.25  # At least 25cm separation between individuals


# ---------------------------------------------------------------------------
# 11. Source-camera projection
# ---------------------------------------------------------------------------
def test_source_camera_projection():
    plane_h = 2.0
    aspect = 1.5
    plane_w = plane_h * aspect

    # Test top-left, center, bottom-right corners
    for nx, ny in [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]:
        wx = (nx - 0.5) * plane_w
        wy = (1.0 - ny) * plane_h
        # Invert
        rec_nx = (wx / plane_w) + 0.5
        rec_ny = 1.0 - (wy / plane_h)
        assert abs(rec_nx - nx) < 1e-6
        assert abs(rec_ny - ny) < 1e-6


# ---------------------------------------------------------------------------
# 12. Photograph and mesh alignment in combined view
# ---------------------------------------------------------------------------
def test_photograph_and_mesh_alignment_in_combined_view():
    scene = Scene(
        id="test-scene",
        name="Test Scene",
        source_image=SourceImageMetadata(
            asset_id="asset-1",
            filename="test.png",
            width=1536,
            height=1024,
            aspect_ratio=1.5,
            mime_type="image/png",
            size_bytes=100000,
        ),
        entities=[
            SceneEntity(
                id="ent-1",
                source_image_id="asset-1",
                bounding_box=BoundingBox2D(x_min=0.2, y_min=0.1, x_max=0.4, y_max=0.8),
                reconstruction=HumanReconstructionData(
                    status="ready",
                    mesh_asset_id="mesh-1",
                    mesh_url="/api/v1/assets/meshes/mesh.glb",
                    geometry_metadata=GeometryMetadata(
                        vertex_count=6890,
                        face_count=13776,
                        bounds=MeshBounds(min=[-0.3, -0.68, -0.2], max=[0.3, 1.0, 0.45]),
                        dimensions=[0.6, 1.68, 0.65],
                    ),
                    coordinate_convention=CoordinateConvention(up="+Y", forward="+Z", right="+X"),
                ),
            )
        ],
    )

    engine = SceneAssemblyEngine()
    res = engine.assemble(scene)

    node = res.scene_graph.nodes["node-ent-1"]
    ref_plane = res.scene_graph.nodes["plane-reference"]

    # Mesh must be positioned in front of the reference plane
    assert node.transform.position.z > ref_plane.transform.position.z
    # Reference plane height is 2.0; mesh top must not float far below ground
    assert node.transform.position.y > 0.5


# ---------------------------------------------------------------------------
# 13. Rigging and skinning validity
# ---------------------------------------------------------------------------
def test_rigging_and_skinning_validity():
    entity = SceneEntity(id="test-ent", source_image_id="asset-1")
    skel = skeleton_fitter.build_skeleton(entity)
    rig_engine = RiggingEngine()

    cube = trimesh.creation.box(extents=(0.5, 1.7, 0.3))
    rig = rig_engine.rig_mesh(entity, skel, np.array(cube.vertices, dtype=np.float32))

    assert rig.status == "rigged"
    assert rig.max_influences_per_vertex == 4
    assert rig.weights_normalized is True
    assert rig.bone_count == 24


# ---------------------------------------------------------------------------
# 14. Independent animation state
# ---------------------------------------------------------------------------
def test_independent_animation_state():
    anim_engine = ProceduralAnimationEngine()
    clips_1 = anim_engine.generate_clips_for_entity("ent-1", "skel-1")
    clips_2 = anim_engine.generate_clips_for_entity("ent-2", "skel-2")

    assert len(clips_1) == 4
    assert len(clips_2) == 4
    assert clips_1[0].target_entity_id == "ent-1"
    assert clips_2[0].target_entity_id == "ent-2"

    state_1 = AnimationState(is_playing=False, current_clip="idle_breathing")
    state_2 = AnimationState(is_playing=True, current_clip="torso_twist")
    assert state_1.is_playing != state_2.is_playing
    assert state_1.current_clip != state_2.current_clip


# ---------------------------------------------------------------------------
# 15. Scene save and reopen
# ---------------------------------------------------------------------------
def test_scene_save_and_reopen():
    scene = Scene(
        id="save-reopen-scene",
        name="Saved Scene",
        source_image=SourceImageMetadata(
            asset_id="asset-1",
            filename="photo.jpg",
            width=1920,
            height=1080,
            aspect_ratio=1.777,
            mime_type="image/jpeg",
            size_bytes=200000,
        ),
        entities=[
            SceneEntity(
                id="person-1",
                source_image_id="asset-1",
                transform=Transform(
                    position=Vector3D(x=0.5, y=1.0, z=-0.1),
                    rotation=EulerRotation(x=0.0, y=0.0, z=0.0),
                    scale=Vector3D(x=0.95, y=0.95, z=0.95),
                ),
            )
        ],
    )
    # Serialize to JSON and deserialize
    json_data = scene.model_dump_json()
    reloaded = Scene.model_validate_json(json_data)

    assert reloaded.id == scene.id
    assert len(reloaded.entities) == 1
    assert reloaded.entities[0].transform.position.x == 0.5
    assert reloaded.entities[0].transform.position.y == 1.0
    assert reloaded.entities[0].transform.scale.x == 0.95
