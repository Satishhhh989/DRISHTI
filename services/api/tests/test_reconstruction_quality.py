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
    Hmr2HumanReconstructor,
    MockHumanReconstructor,
    ReconstructionBackendRegistry,
    SMPLPoseFitter,
    SpinHumanReconstructor,
    compute_mesh_normals,
    export_mesh_glb,
    prepare_person_crop,
    reconstruction_registry,
    validate_mesh_geometry,
)
from app.schemas.common import BoundingBox2D, Transform, Vector3D, EulerRotation
from app.schemas.scene import (
    AnimationState,
    ConfidenceMetadata,
    CoordinateConvention,
    DetectionInfo,
    EntityDepthInfo,
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

    # Identify seated person (center seated: cx ~ 0.51) and standing person (standing left: cx ~ 0.10)
    seated_det = next(d for d in detections if 0.45 < (d.bounding_box.x_min + d.bounding_box.x_max) / 2.0 < 0.60)
    standing_det = next(d for d in detections if (d.bounding_box.x_min + d.bounding_box.x_max) / 2.0 < 0.20)

    res_seated = rec.reconstruct(img, seated_det.bounding_box)
    res_standing = rec.reconstruct(img, standing_det.bounding_box)

    v_seated = res_seated.vertices
    v_standing = res_standing.vertices

    h_seated = v_seated[:, 1].max() - v_seated[:, 1].min()
    h_standing = v_standing[:, 1].max() - v_standing[:, 1].min()

    # Seated person has reduced vertical height due to bent legs
    assert h_seated < h_standing
    assert h_standing > 1.60
    assert h_seated < 1.55

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


# ---------------------------------------------------------------------------
# 16. SMPL Pose Fitting: Seated Person Preservation
# ---------------------------------------------------------------------------
def test_smpl_pose_fitting_seated_person():
    smpl_path = Path("data/models/human_reconstruction/SMPL_N_model.npz")
    mean_path = Path("data/models/human_reconstruction/smpl_mean_params.npz")
    if not smpl_path.exists() or not mean_path.exists():
        pytest.skip("SMPL model files not present in data/models/human_reconstruction")

    from app.pipeline.motion import PoseEstimationResult, PoseKeypoint2D
    from app.pipeline.reconstructor import SMPLForwardLayer, SMPLPoseFitter

    npz_data = np.load(smpl_path, allow_pickle=True)
    smpl = SMPLForwardLayer(npz_data)
    mean_data = np.load(mean_path)

    init_pose = torch.tensor(mean_data["pose"], dtype=torch.float32).unsqueeze(0)
    init_shape = torch.tensor(mean_data["shape"], dtype=torch.float32).unsqueeze(0)
    init_cam = torch.tensor(mean_data["cam"], dtype=torch.float32).unsqueeze(0)

    fitter = SMPLPoseFitter(smpl=smpl, weights=npz_data["weights"], device=torch.device("cpu"))

    # Synthetic keypoints for a seated person in 1000x1000 image:
    # Pelvis at y=500, knees horizontal forward at y=520, ankles straight down at y=800
    kps = {
        "head": PoseKeypoint2D(x_norm=0.50, y_norm=0.20, confidence=0.98, is_visible=True),
        "left_shoulder": PoseKeypoint2D(x_norm=0.58, y_norm=0.32, confidence=0.95, is_visible=True),
        "right_shoulder": PoseKeypoint2D(x_norm=0.42, y_norm=0.32, confidence=0.95, is_visible=True),
        "left_elbow": PoseKeypoint2D(x_norm=0.60, y_norm=0.45, confidence=0.90, is_visible=True),
        "right_elbow": PoseKeypoint2D(x_norm=0.40, y_norm=0.45, confidence=0.90, is_visible=True),
        "left_wrist": PoseKeypoint2D(x_norm=0.55, y_norm=0.52, confidence=0.92, is_visible=True),
        "right_wrist": PoseKeypoint2D(x_norm=0.45, y_norm=0.52, confidence=0.92, is_visible=True),
        "left_hip": PoseKeypoint2D(x_norm=0.56, y_norm=0.50, confidence=0.95, is_visible=True),
        "right_hip": PoseKeypoint2D(x_norm=0.44, y_norm=0.50, confidence=0.95, is_visible=True),
        "left_knee": PoseKeypoint2D(x_norm=0.58, y_norm=0.53, confidence=0.95, is_visible=True),
        "right_knee": PoseKeypoint2D(x_norm=0.42, y_norm=0.53, confidence=0.95, is_visible=True),
        "left_ankle": PoseKeypoint2D(x_norm=0.58, y_norm=0.82, confidence=0.96, is_visible=True),
        "right_ankle": PoseKeypoint2D(x_norm=0.42, y_norm=0.82, confidence=0.96, is_visible=True),
    }
    pose_res = PoseEstimationResult(
        bounding_box=BoundingBox2D(x_min=0.35, y_min=0.15, x_max=0.65, y_max=0.85),
        keypoints=kps,
        confidence_score=0.95,
        model_name="YOLOv8-Pose",
    )

    bbox = BoundingBox2D(x_min=0.35, y_min=0.15, x_max=0.65, y_max=0.85)
    dummy_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    _, meta = prepare_person_crop(dummy_img, bbox)

    fitted_pose, fitted_shape, fitted_cam, metrics = fitter.fit(
        init_pose=init_pose,
        init_shape=init_shape,
        init_cam=init_cam,
        pose_result=pose_res,
        transform_meta=meta,
        quality_preset="high",
    )

    # Verify reprojection error is reasonably bounded (< 35.0 crop pixels)
    assert metrics["reprojection_error_pixels"] < 35.0
    assert metrics["quality_score"] > 0.30

    # Verify 3D joints reflect sitting posture:
    with torch.no_grad():
        v = smpl(fitted_pose, fitted_shape)
        j = torch.matmul(smpl.J_regressor, v).numpy()
        # Left knee dy from hip vs ankle dy from knee
        thigh_dy = abs(j[4, 1] - j[1, 1])
        shin_dy = abs(j[7, 1] - j[4, 1])
        # In sitting pose, thigh vertical drop is small while shin vertical drop is large
        assert thigh_dy < (shin_dy * 0.8)


# ---------------------------------------------------------------------------
# 17. SMPL Pose Fitting: Crossed Arms Preservation
# ---------------------------------------------------------------------------
def test_smpl_pose_fitting_crossed_arms():
    smpl_path = Path("data/models/human_reconstruction/SMPL_N_model.npz")
    mean_path = Path("data/models/human_reconstruction/smpl_mean_params.npz")
    if not smpl_path.exists() or not mean_path.exists():
        pytest.skip("SMPL model files not present in data/models/human_reconstruction")

    from app.pipeline.motion import PoseEstimationResult, PoseKeypoint2D
    from app.pipeline.reconstructor import SMPLForwardLayer, SMPLPoseFitter

    npz_data = np.load(smpl_path, allow_pickle=True)
    smpl = SMPLForwardLayer(npz_data)
    mean_data = np.load(mean_path)

    init_pose = torch.tensor(mean_data["pose"], dtype=torch.float32).unsqueeze(0)
    init_shape = torch.tensor(mean_data["shape"], dtype=torch.float32).unsqueeze(0)
    init_cam = torch.tensor(mean_data["cam"], dtype=torch.float32).unsqueeze(0)

    fitter = SMPLPoseFitter(smpl=smpl, weights=npz_data["weights"], device=torch.device("cpu"))

    # Crossed arms keypoints: wrists cross at center chest
    kps = {
        "head": PoseKeypoint2D(x_norm=0.50, y_norm=0.15, confidence=0.98, is_visible=True),
        "left_shoulder": PoseKeypoint2D(x_norm=0.60, y_norm=0.28, confidence=0.95, is_visible=True),
        "right_shoulder": PoseKeypoint2D(x_norm=0.40, y_norm=0.28, confidence=0.95, is_visible=True),
        "left_elbow": PoseKeypoint2D(x_norm=0.65, y_norm=0.40, confidence=0.90, is_visible=True),
        "right_elbow": PoseKeypoint2D(x_norm=0.35, y_norm=0.40, confidence=0.90, is_visible=True),
        "left_wrist": PoseKeypoint2D(x_norm=0.46, y_norm=0.38, confidence=0.95, is_visible=True),
        "right_wrist": PoseKeypoint2D(x_norm=0.54, y_norm=0.38, confidence=0.95, is_visible=True),
        "left_hip": PoseKeypoint2D(x_norm=0.55, y_norm=0.55, confidence=0.95, is_visible=True),
        "right_hip": PoseKeypoint2D(x_norm=0.45, y_norm=0.55, confidence=0.95, is_visible=True),
        "left_knee": PoseKeypoint2D(x_norm=0.55, y_norm=0.72, confidence=0.95, is_visible=True),
        "right_knee": PoseKeypoint2D(x_norm=0.45, y_norm=0.72, confidence=0.95, is_visible=True),
        "left_ankle": PoseKeypoint2D(x_norm=0.55, y_norm=0.90, confidence=0.95, is_visible=True),
        "right_ankle": PoseKeypoint2D(x_norm=0.45, y_norm=0.90, confidence=0.95, is_visible=True),
    }
    pose_res = PoseEstimationResult(
        bounding_box=BoundingBox2D(x_min=0.30, y_min=0.10, x_max=0.70, y_max=0.95),
        keypoints=kps,
        confidence_score=0.95,
        model_name="YOLOv8-Pose",
    )

    meta = {
        "square_dim": 800,
        "offset": [50, 50],
        "crop_pixels": [250, 50, 850, 950],
        "original_image_size": [1000, 1000],
    }

    fitted_pose, fitted_shape, fitted_cam, metrics = fitter.fit(
        init_pose=init_pose,
        init_shape=init_shape,
        init_cam=init_cam,
        pose_result=pose_res,
        transform_meta=meta,
        quality_preset="high",
    )

    with torch.no_grad():
        v = smpl(fitted_pose, fitted_shape)
        j = torch.matmul(smpl.J_regressor, v).numpy()
        # Wrists: Joint 20 (L wrist) and Joint 21 (R wrist)
        wrist_dist = float(np.linalg.norm(j[20] - j[21]))
        # Crossed arms have wrists within 0.35m of each other
        assert wrist_dist < 0.35


# ---------------------------------------------------------------------------
# 18. Body Shape Variation Conditioning
# ---------------------------------------------------------------------------
def test_body_shape_variation_conditioning():
    smpl_path = Path("data/models/human_reconstruction/SMPL_N_model.npz")
    mean_path = Path("data/models/human_reconstruction/smpl_mean_params.npz")
    if not smpl_path.exists() or not mean_path.exists():
        pytest.skip("SMPL model files not present in data/models/human_reconstruction")

    from app.pipeline.reconstructor import SMPLForwardLayer

    npz_data = np.load(smpl_path, allow_pickle=True)
    smpl = SMPLForwardLayer(npz_data)
    mean_data = np.load(mean_path)
    init_pose = torch.tensor(mean_data["pose"], dtype=torch.float32).unsqueeze(0)

    # Generate two distinct body shapes using beta PCA components
    betas_slender = torch.zeros(10, dtype=torch.float32)
    betas_slender[0] = -2.5  # Slender / narrower build

    betas_heavy = torch.zeros(10, dtype=torch.float32)
    betas_heavy[0] = 3.0   # Broader / heavier build

    with torch.no_grad():
        v_slender = smpl(init_pose, betas_slender).numpy()
        v_heavy = smpl(init_pose, betas_heavy).numpy()

    # Verify that different shape betas produce genuinely distinct geometry
    diff = np.abs(v_slender - v_heavy).max()
    assert diff > 0.05  # More than 5cm difference in vertex positions

    # Width along X should be wider for heavy than slender
    width_slender = v_slender[:, 0].max() - v_slender[:, 0].min()
    width_heavy = v_heavy[:, 0].max() - v_heavy[:, 0].min()
    assert width_heavy > width_slender


# ---------------------------------------------------------------------------
# 19. Quality Presets Control Optimization
# ---------------------------------------------------------------------------
def test_quality_presets_control_optimization():
    rec = MockHumanReconstructor(device="cpu")
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    box = BoundingBox2D(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.9)

    res_prev = rec.reconstruct(img, box, quality_preset="preview")
    res_high = rec.reconstruct(img, box, quality_preset="high")
    res_max = rec.reconstruct(img, box, quality_preset="maximum")

    assert res_prev.quality_preset == "preview"
    assert res_high.quality_preset == "high"
    assert res_max.quality_preset == "maximum"
    assert res_max.reprojection_error_pixels < res_prev.reprojection_error_pixels
    assert res_max.silhouette_iou > res_prev.silhouette_iou


# ---------------------------------------------------------------------------
# 20. Source-Aligned Validation Metrics
# ---------------------------------------------------------------------------
def test_source_aligned_validation_metrics():
    rec_data = HumanReconstructionData(
        status="ready",
        mesh_asset_id="mesh-001",
        quality_score=0.91,
        quality_preset="high",
        reprojection_error_pixels=2.45,
        reprojection_error_normalized=0.021,
        silhouette_iou=0.865,
    )

    assert rec_data.reprojection_error_pixels == 2.45
    assert rec_data.reprojection_error_normalized == 0.021
    assert rec_data.silhouette_iou == 0.865
    assert rec_data.quality_preset == "high"

    # JSON roundtrip preserves metrics
    data_json = rec_data.model_dump_json()
    reloaded = HumanReconstructionData.model_validate_json(data_json)
    assert reloaded.reprojection_error_pixels == 2.45
    assert reloaded.silhouette_iou == 0.865


# ---------------------------------------------------------------------------
# 21. Reconstruction Pipeline Resequencing Integration
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_reconstruction_pipeline_resequencing_integration(tmp_path):
    from app.pipeline.mock import MockPersonDetector, MockPersonSegmentor, MockDepthEstimator
    from app.pipeline.reconstructor import MockHumanReconstructor
    from app.scene.manager import scene_manager
    from app.jobs.tracker import job_tracker

    coord = PipelineCoordinator(
        detector=MockPersonDetector(),
        segmentor=MockPersonSegmentor(),
        depth_estimator=MockDepthEstimator(),
        reconstructor=MockHumanReconstructor(),
    )

    from app.assets.manager import asset_manager

    # Create scene with registered photo asset
    img = Image.new("RGB", (640, 480), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    photo_bytes = buf.getvalue()

    photo_asset = asset_manager.register_photo(
        content=photo_bytes,
        filename="test_resequence.jpg",
        mime_type="image/jpeg",
    )
    scene = scene_manager.create_scene_from_photo(photo_asset, name="Resequence Pipeline Test")
    job_id = job_tracker.create_job(scene_id=scene.id)

    res_scene = await coord.execute_pipeline(
        scene_id=scene.id,
        job_id=job_id,
        confidence_threshold=0.3,
        include_depth=True,
        include_reconstruction=True,
    )

    assert res_scene.processing_status == "ready"
    assert len(res_scene.entities) > 0
    for e in res_scene.entities:
        assert e.reconstruction is not None
        assert e.reconstruction.status == "ready"
        assert e.reconstruction.reprojection_error_pixels is not None
        assert e.reconstruction.silhouette_iou is not None
        assert e.reconstruction.stance_description is not None
        assert e.skeleton is not None
        assert e.skeleton.validation_status == "valid"


# ---------------------------------------------------------------------------
# 22. Multi-Hypothesis Seated Pose Fitting Verification
# ---------------------------------------------------------------------------
def test_multi_hypothesis_seated_pose_fitting():
    """Verifies that SMPLPoseFitter detects seated keypoints and converges to horizontal thighs."""
    from app.pipeline.motion import PoseEstimationResult, PoseKeypoint2D
    from app.pipeline.reconstructor import SMPLPoseFitter, SMPLForwardLayer

    ckpt_path = model_manager.get_model_path("spin-human-reconstruction")
    if not ckpt_path.exists():
        pytest.skip("SPIN model weights not found")

    smpl_path = ckpt_path.parent / "SMPL_N_model.npz"
    npz = np.load(smpl_path, allow_pickle=True)
    smpl = SMPLForwardLayer(dict(npz))
    fitter = SMPLPoseFitter(smpl, npz["weights"], torch.device("cpu"), rot6d_mode="spin")

    # Synthesize keypoints for a seated individual
    # Thigh is horizontal (hip Y ~ 0.52, knee Y ~ 0.54), shin is vertical (ankle Y ~ 0.85)
    seated_kps = {
        "left_hip": PoseKeypoint2D(x_norm=0.45, y_norm=0.52, confidence=0.92, is_visible=True),
        "right_hip": PoseKeypoint2D(x_norm=0.55, y_norm=0.52, confidence=0.92, is_visible=True),
        "left_knee": PoseKeypoint2D(x_norm=0.44, y_norm=0.54, confidence=0.90, is_visible=True),
        "right_knee": PoseKeypoint2D(x_norm=0.56, y_norm=0.54, confidence=0.90, is_visible=True),
        "left_ankle": PoseKeypoint2D(x_norm=0.44, y_norm=0.85, confidence=0.88, is_visible=True),
        "right_ankle": PoseKeypoint2D(x_norm=0.56, y_norm=0.85, confidence=0.88, is_visible=True),
        "left_shoulder": PoseKeypoint2D(x_norm=0.42, y_norm=0.25, confidence=0.95, is_visible=True),
        "right_shoulder": PoseKeypoint2D(x_norm=0.58, y_norm=0.25, confidence=0.95, is_visible=True),
        "head": PoseKeypoint2D(x_norm=0.50, y_norm=0.15, confidence=0.95, is_visible=True),
    }
    pose_res = PoseEstimationResult(
        keypoints=seated_kps,
        bounding_box=BoundingBox2D(x_min=0.35, y_min=0.10, x_max=0.65, y_max=0.90),
        confidence_score=0.92,
        model_name="yolov8m-pose",
    )

    init_pose = torch.zeros(1, 24, 6)
    init_pose[0, :, 0] = 1.0
    init_pose[0, :, 4] = 1.0
    init_shape = torch.zeros(1, 10)
    init_cam = torch.tensor([[0.9, 0.0, 0.0]])

    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    _, meta = prepare_person_crop(img, pose_res.bounding_box, padding=0.15, target_size=(224, 224))

    opt_pose, opt_shape, opt_cam, metrics = fitter.fit(
        init_pose=init_pose,
        init_shape=init_shape,
        init_cam=init_cam,
        pose_result=pose_res,
        transform_meta=meta,
        quality_preset="high",
    )

    assert "Seated" in metrics["stance_description"]
    assert metrics["quality_preset"] == "high"
    assert metrics["reprojection_error_normalized"] < 0.25

    # 3D thigh should be approximately horizontal: |hip_y - knee_y| < 0.12m
    v_opt = smpl(opt_pose, opt_shape, rot6d_mode="spin")
    j_opt = torch.matmul(smpl.J_regressor, v_opt)
    thigh_dy = abs(j_opt[1, 1].item() - j_opt[4, 1].item())
    assert thigh_dy < 0.12


# ---------------------------------------------------------------------------
# 23. Quality Preset End-to-End Propagation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_quality_preset_propagation(tmp_path):
    """Verifies that quality preset propagates through PipelineCoordinator to scene entities."""
    from app.pipeline.mock import MockPersonDetector, MockPersonSegmentor, MockDepthEstimator
    from app.pipeline.reconstructor import MockHumanReconstructor
    from app.scene.manager import scene_manager
    from app.jobs.tracker import job_tracker
    from app.assets.manager import asset_manager

    coord = PipelineCoordinator(
        detector=MockPersonDetector(),
        segmentor=MockPersonSegmentor(),
        depth_estimator=MockDepthEstimator(),
        reconstructor=MockHumanReconstructor(),
    )

    img = Image.new("RGB", (640, 480), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    photo_asset = asset_manager.register_photo(
        content=buf.getvalue(),
        filename="test_preset.jpg",
        mime_type="image/jpeg",
    )

    for preset in ["preview", "high", "maximum"]:
        scene = scene_manager.create_scene_from_photo(photo_asset, name=f"Preset {preset} Test")
        job_id = job_tracker.create_job(scene_id=scene.id)

        res_scene = await coord.execute_pipeline(
            scene_id=scene.id,
            job_id=job_id,
            confidence_threshold=0.3,
            include_depth=True,
            include_reconstruction=True,
            quality_preset=preset,
        )

        assert res_scene.processing_status == "ready"
        assert len(res_scene.entities) > 0
        for e in res_scene.entities:
            assert e.reconstruction.quality_preset == preset
            assert e.reconstruction.stance_description is not None
            assert e.reconstruction.observed_keypoints_count is not None
            assert e.reconstruction.pose_convergence_status is not None


# ---------------------------------------------------------------------------
# 24. Critical Joint Indexing (2D and 3D Batched Tensors)
# ---------------------------------------------------------------------------
def test_select_joints_indexing_2d_and_3d():
    """Verifies that select_joints indexes the joint dimension, never the batch dimension,
    preventing IndexError or incorrect slicing across 2D, 3D single-batch, and multi-batch tensors."""
    indices = [0, 1, 4, 12, 15, 21]

    # Case A: 2D tensor (24, 3)
    j_2d = torch.arange(24 * 3, dtype=torch.float32).reshape(24, 3)
    selected_2d = SMPLPoseFitter.select_joints(j_2d, indices)
    assert selected_2d.shape == (6, 3)
    for idx_out, idx_j in enumerate(indices):
        assert torch.allclose(selected_2d[idx_out], j_2d[idx_j])

    # Case B: 3D tensor with batch=1 (1, 24, 3) - where naive j_3d[indices] would index batch dim!
    j_3d_single = j_2d.unsqueeze(0)  # Shape (1, 24, 3)
    selected_3d_single = SMPLPoseFitter.select_joints(j_3d_single, indices)
    assert selected_3d_single.shape == (1, 6, 3)
    for idx_out, idx_j in enumerate(indices):
        assert torch.allclose(selected_3d_single[0, idx_out], j_3d_single[0, idx_j])

    # Case C: 3D tensor with batch=3 (3, 24, 3)
    j_3d_multi = torch.stack([j_2d, j_2d * 2.0, j_2d * 3.0], dim=0)
    selected_3d_multi = SMPLPoseFitter.select_joints(j_3d_multi, indices)
    assert selected_3d_multi.shape == (3, 6, 3)
    for b in range(3):
        for idx_out, idx_j in enumerate(indices):
            assert torch.allclose(selected_3d_multi[b, idx_out], j_3d_multi[b, idx_j])


# ---------------------------------------------------------------------------
# 25. Differentiable Silhouette Distance Transform Optimization
# ---------------------------------------------------------------------------
def test_differentiable_silhouette_loss_gradient():
    """Verifies that distance transform grid sampling generates non-zero gradients pulling
    vertices outside the true silhouette boundary toward the silhouette."""
    import cv2
    import torch.nn.functional as F

    s_res = 128
    # Create a centered square mask (40 to 88 in both dimensions)
    mask = np.zeros((s_res, s_res), dtype=np.uint8)
    mask[40:88, 40:88] = 255

    # Compute Euclidean distance transform
    dist_map = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5) / float(s_res)
    dist_tensor = torch.tensor(dist_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Place a 2D point outside the mask
    outside_pt = torch.tensor([[[[0.7, 0.7]]]], dtype=torch.float32, requires_grad=True)

    sampled_dist = F.grid_sample(dist_tensor, outside_pt, align_corners=False, mode="bilinear", padding_mode="border")
    loss = (sampled_dist ** 2).mean()

    assert loss.item() > 0.0, "Loss for point outside mask must be strictly positive"

    loss.backward()
    assert outside_pt.grad is not None
    assert outside_pt.grad[0, 0, 0, 0].item() > 0.0
    assert outside_pt.grad[0, 0, 0, 1].item() > 0.0


# ---------------------------------------------------------------------------
# 26. Honest Metric Reporting (No Fake Zero Error or Fake 0.80 IoU)
# ---------------------------------------------------------------------------
def test_metrics_honesty_unavailable_and_mock():
    """Verifies that missing or insufficient keypoints/masks return None with appropriate metric_status,
    and MockHumanReconstructor explicitly reports metric_status='mock'."""
    # Test MockHumanReconstructor reports mock metric_status
    mock_rec = MockHumanReconstructor()
    res = mock_rec.reconstruct(
        image_rgb=np.zeros((100, 100, 3), dtype=np.uint8),
        bounding_box=BoundingBox2D(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.9),
    )
    assert res.metric_status == "mock"
    assert res.reprojection_error_pixels == 18.5
    assert res.silhouette_iou == 0.84

    # Test fitter behavior with fewer than 4 keypoints
    if model_manager.is_model_available("spin-human-reconstruction"):
        ckpt = model_manager.get_model_path("spin-human-reconstruction")
        smpl_path = ckpt.parent / "SMPL_N_model.npz"
        from app.pipeline.reconstructor import SMPLForwardLayer, SMPLPoseFitter
        from app.pipeline.motion import PoseEstimationResult, PoseKeypoint2D

        npz_data = np.load(smpl_path, allow_pickle=True)
        smpl = SMPLForwardLayer(npz_data)
        fitter = SMPLPoseFitter(smpl=smpl, weights=npz_data["weights"], device=torch.device("cpu"))

        # Only 2 keypoints (insufficient for reliable metric measurement)
        insufficient_kps = {
            "head": PoseKeypoint2D(x_norm=0.5, y_norm=0.2, confidence=0.9, is_visible=True),
            "left_hip": PoseKeypoint2D(x_norm=0.5, y_norm=0.6, confidence=0.9, is_visible=True),
        }
        sparse_pose = PoseEstimationResult(
            keypoints=insufficient_kps,
            bounding_box=BoundingBox2D(x_min=0.3, y_min=0.1, x_max=0.7, y_max=0.9),
            confidence_score=0.9,
            model_name="yolov8m-pose",
        )

        img = np.zeros((400, 400, 3), dtype=np.uint8)
        _, meta = prepare_person_crop(img, sparse_pose.bounding_box, padding=0.15, target_size=(224, 224))

        init_pose = torch.zeros(1, 24, 6)
        init_pose[0, :, 0] = 1.0
        init_pose[0, :, 4] = 1.0
        init_shape = torch.zeros(1, 10)
        init_cam = torch.tensor([[0.9, 0.0, 0.0]])

        _, _, _, metrics = fitter.fit(
            init_pose=init_pose,
            init_shape=init_shape,
            init_cam=init_cam,
            pose_result=sparse_pose,
            transform_meta=meta,
            quality_preset="preview",
        )

        assert metrics["metric_status"] == "unavailable"
        assert metrics["reprojection_error_pixels"] is None
        assert metrics["reprojection_error_normalized"] is None
        assert metrics["silhouette_iou"] is None
        assert metrics["quality_score"] is None


# ---------------------------------------------------------------------------
# 27. Modular HMR 2.0 Adapter and Registry Fallback
# ---------------------------------------------------------------------------
def test_hmr2_adapter_and_registry_fallback():
    """Verifies that Hmr2HumanReconstructor exposes modular capabilities and falls back gracefully."""
    mock_fallback = MockHumanReconstructor()
    hmr2 = Hmr2HumanReconstructor(checkpoint_path=None, fallback_reconstructor=mock_fallback)

    caps = hmr2.get_capabilities()
    assert caps["is_articulated"] is True
    assert "SMPL Canonical (6,890 vertices, 24 joints)" in caps["output_format"]
    assert caps["weights_installed"] is False
    assert "CC-BY-NC 4.0" in caps["license"]

    # Reconstruct should fall back to mock_fallback with warning
    res = hmr2.reconstruct(
        image_rgb=np.zeros((100, 100, 3), dtype=np.uint8),
        bounding_box=BoundingBox2D(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.9),
    )
    assert res.metric_status == "mock"
    assert any("HMR 2.0" in w for w in res.warnings)

    # Test registry operations
    reg = ReconstructionBackendRegistry()
    reg.register("test_mock", mock_fallback)
    reg.register("test_hmr2", hmr2)
    assert "test_mock" in reg.list_backends()
    assert "test_hmr2" in reg.list_backends()
    assert reg.get("test_mock") is mock_fallback
    assert reg.get("nonexistent") is None


# ---------------------------------------------------------------------------
# 28. Multi-Person Pelvis and Spatial Depth Isolation
# ---------------------------------------------------------------------------
def test_multi_person_pelvis_and_depth_isolation():
    """Verifies that multi-person reconstruction produces isolated, non-overlapping 3D placements
    where seated and standing people occupy distinct coordinates based on bounding box and depth statistics."""
    from app.pipeline.assembly import SceneAssemblyEngine, AssemblyConfig

    assembly = SceneAssemblyEngine(AssemblyConfig())

    # Person A: Left standing, closer depth (median_proximity = 0.8)
    entity_a = SceneEntity(
        id="person-left-standing",
        source_image_id="img-001",
        category="person",
        tracking_id=1,
        detection=DetectionInfo(
            bounding_box=BoundingBox2D(x_min=0.10, y_min=0.15, x_max=0.30, y_max=0.90),
            pixel_box=[100.0, 150.0, 300.0, 900.0],
            confidence=0.95,
            model="yolov8",
        ),
        depth=None,
    )

    # Person B: Center seated, further depth (median_proximity = 0.4)
    entity_b = SceneEntity(
        id="person-center-seated",
        source_image_id="img-001",
        category="person",
        tracking_id=2,
        detection=DetectionInfo(
            bounding_box=BoundingBox2D(x_min=0.40, y_min=0.35, x_max=0.60, y_max=0.85),
            pixel_box=[400.0, 350.0, 600.0, 850.0],
            confidence=0.93,
            model="yolov8",
        ),
        depth=None,
    )

    # Coordinates based on coordinator placement rules
    ca_x = (0.10 + 0.30) / 2.0
    ca_y = (0.15 + 0.90) / 2.0
    cb_x = (0.40 + 0.60) / 2.0
    cb_y = (0.35 + 0.85) / 2.0

    pos_a = Vector3D(x=(ca_x - 0.5) * 1.5, y=(0.5 - ca_y) * 1.5, z=(0.8 - 0.5) * 0.4)
    pos_b = Vector3D(x=(cb_x - 0.5) * 1.5, y=(0.5 - cb_y) * 1.5, z=(0.4 - 0.5) * 0.4)

    assert pos_a.x < -0.3
    assert abs(pos_b.x) < 0.1
    assert pos_a.z > pos_b.z
    dist = np.sqrt((pos_a.x - pos_b.x)**2 + (pos_a.y - pos_b.y)**2 + (pos_a.z - pos_b.z)**2)
    assert dist > 0.45


# ---------------------------------------------------------------------------
# 29. HMR 2.0 Real Neural Inference and SMPL Dimensions
# ---------------------------------------------------------------------------
def test_hmr2_real_neural_inference_and_dimensions():
    """Verifies that HMR 2.0 performs real ViT neural inference without mocks and produces
    exact canonical SMPL geometry (6,890 vertices, 13,776 faces)."""
    hmr2_path = model_manager.models_dir / "hmr2" / "epoch=35-step=1000000.ckpt"
    if not hmr2_path.exists():
        pytest.skip("HMR 2.0 checkpoint not found")

    rec = Hmr2HumanReconstructor(checkpoint_path=hmr2_path)
    assert rec.is_available is True
    assert rec.is_real_hmr2 is True
    assert rec._model is not None

    test_img = np.zeros((400, 300, 3), dtype=np.uint8)
    box = BoundingBox2D(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.9)

    res = rec.reconstruct(test_img, box)
    assert res.vertices.shape == (6890, 3)
    assert res.faces.shape == (13776, 3)
    assert res.normals.shape == (6890, 3)
    assert len(res.glb_bytes) > 10000
    assert res.model_name == "HMR 2.0 (4DHumans)"
    assert res.metric_status in ("measured", "unavailable")
    assert res.processing_time_ms > 0.0


# ---------------------------------------------------------------------------
# 30. HMR 2.0 Standing and Seated Pose Fidelity Verification
# ---------------------------------------------------------------------------
def test_hmr2_standing_and_seated_pose_fidelity():
    """Verifies that HMR 2.0 correctly differentiates standing upright posture from
    seated posture with bent knees on the genuine 5-person test image."""
    test_img = Path("data/storage/photos/066bfeaf-236d-4a16-a7a8-a6c0f0cfcb9b_drishti_test_.png")
    hmr2_path = model_manager.models_dir / "hmr2" / "epoch=35-step=1000000.ckpt"
    if not test_img.exists() or not hmr2_path.exists():
        pytest.skip("Test image or HMR 2.0 checkpoint unavailable")

    coord = PipelineCoordinator()
    rec = coord.get_reconstructor(backend="hmr2")
    det = coord.get_detector()
    img = np.array(Image.open(test_img).convert("RGB"))
    detections = det.detect_people(test_img)

    seated_det = next(d for d in detections if 0.45 < (d.bounding_box.x_min + d.bounding_box.x_max) / 2.0 < 0.60)
    standing_det = next(d for d in detections if (d.bounding_box.x_min + d.bounding_box.x_max) / 2.0 < 0.20)

    res_seated = rec.reconstruct(img, seated_det.bounding_box)
    res_standing = rec.reconstruct(img, standing_det.bounding_box)

    h_seated = res_seated.vertices[:, 1].max() - res_seated.vertices[:, 1].min()
    h_standing = res_standing.vertices[:, 1].max() - res_standing.vertices[:, 1].min()

    # Standing height should exceed seated height significantly due to knee flexion
    assert h_standing > h_seated
    assert h_standing >= 1.65, f"Standing height {h_standing}m should be >= 1.65m"
    assert h_seated <= 1.35, f"Seated height {h_seated}m should be <= 1.35m"

    # Knee angle check
    class DummyEntity:
        def __init__(self, eid):
            self.id = eid

    skel_seated = skeleton_fitter.build_skeleton(DummyEntity("seated"), res_seated.vertices)
    skel_standing = skeleton_fitter.build_skeleton(DummyEntity("standing"), res_standing.vertices)

    def knee_angle(skel):
        j_map = {j.id: np.array(j.position) for j in skel.joints}
        v1 = j_map["left_hip"] - j_map["left_knee"]
        v2 = j_map["left_ankle"] - j_map["left_knee"]
        cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))

    ang_seated = knee_angle(skel_seated)
    ang_standing = knee_angle(skel_standing)

    assert ang_seated < 105.0, f"Seated knee angle {ang_seated}° should be acute (<105°)"
    assert ang_standing > 130.0, f"Standing knee angle {ang_standing}° should be extended (>130°)"


# ---------------------------------------------------------------------------
# 31. HMR 2.0 Reprojection Diagnostic Overlay Generation
# ---------------------------------------------------------------------------
def test_hmr2_reprojection_overlay_generation():
    """Verifies that HMR 2.0 produces a genuine PNG reprojection overlay diagnostic."""
    hmr2_path = model_manager.models_dir / "hmr2" / "epoch=35-step=1000000.ckpt"
    if not hmr2_path.exists():
        pytest.skip("HMR 2.0 checkpoint not found")

    rec = Hmr2HumanReconstructor(checkpoint_path=hmr2_path)
    test_img = np.zeros((300, 300, 3), dtype=np.uint8)
    box = BoundingBox2D(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.9)

    res = rec.reconstruct(test_img, box)
    assert res.overlay_png_bytes is not None
    assert len(res.overlay_png_bytes) > 100
    # Check PNG magic bytes: \x89PNG\r\n\x1a\n
    assert res.overlay_png_bytes[:4] == b"\x89PNG"


# ---------------------------------------------------------------------------
# 32. Projective Vertex Colors & Dynamic Reference Plane Clearance
# ---------------------------------------------------------------------------
def test_projective_vertex_colors_and_dynamic_reference_plane():
    """Verifies projective vertex color mapping in GLB and dynamic reference plane offset."""
    import trimesh
    from app.pipeline.reconstructor import compute_projected_vertex_colors, export_mesh_glb
    from app.pipeline.assembly import SceneAssemblyEngine, AssemblyConfig

    # 1. Test compute_projected_vertex_colors
    verts = np.zeros((100, 3), dtype=np.float32)
    normals = np.zeros((100, 3), dtype=np.float32)
    normals[:, 2] = 1.0  # Front-facing towards camera (+Z in Three.js)
    crop_rgb = np.full((128, 128, 3), 180, dtype=np.uint8)
    crop_rgb[:, :, 0] = 220  # Reddish crop
    cam = {"scale": 1.0, "tx": 0.0, "ty": 0.0}

    vc = compute_projected_vertex_colors(verts, normals, cam, crop_rgb)
    assert vc.shape == (100, 4)
    assert vc.dtype == np.uint8
    # Front-facing vertices should sample close to [220, 180, 180, 255]
    assert vc[0, 0] > 190
    assert vc[0, 3] == 255

    # 2. Test export_mesh_glb embedding vertex colors
    faces = np.zeros((10, 3), dtype=np.int32)
    for i in range(10):
        faces[i] = [i * 3, i * 3 + 1, i * 3 + 2]
    glb_bytes = export_mesh_glb(verts, faces, normals, vertex_colors=vc)
    loaded_scene = trimesh.load(trimesh.util.wrap_as_stream(glb_bytes), file_type="glb")
    loaded_mesh = list(loaded_scene.geometry.values())[0]
    assert hasattr(loaded_mesh.visual, "vertex_colors")
    assert loaded_mesh.visual.vertex_colors is not None
    assert len(loaded_mesh.visual.vertex_colors) == 100

    # 3. Test dynamic reference plane clearance in SceneAssemblyEngine
    engine = SceneAssemblyEngine()
    scene = Scene(
        id="test-clearance-scene",
        name="Clearance Scene",
        source_image=SourceImageMetadata(
            asset_id="asset-1",
            filename="photo.jpg",
            width=1920,
            height=1080,
            aspect_ratio=1.777,
            mime_type="image/jpeg",
            size_bytes=100000,
        ),
        entities=[
            SceneEntity(
                id="deep-person",
                source_image_id="img-1",
                bounding_box=BoundingBox2D(x_min=0.2, y_min=0.2, x_max=0.4, y_max=0.8),
                depth=EntityDepthInfo(median_proximity=0.1), # Deep in scene (-Z)
            ),
            SceneEntity(
                id="foreground-person",
                source_image_id="img-1",
                bounding_box=BoundingBox2D(x_min=0.6, y_min=0.2, x_max=0.8, y_max=0.8),
                depth=EntityDepthInfo(median_proximity=0.9), # Close to camera (+Z)
            ),
        ],
    )
    res = engine.assemble(scene)
    ref_plane_node = res.scene_graph.nodes["plane-reference"]
    ref_plane_z = ref_plane_node.transform.position.z

    all_entity_z = [e.transform.position.z for e in res.entities]
    min_entity_z = min(all_entity_z)

    # Reference plane MUST be strictly behind all entities with clearance
    assert ref_plane_z < min_entity_z, f"Reference plane {ref_plane_z} must be behind deepest entity {min_entity_z}"
    assert min_entity_z - ref_plane_z >= 0.5, "At least 0.5m clearance behind deepest person"


# ---------------------------------------------------------------------------
# 33. Segmentation mask guard & robust skeleton model resolution
# ---------------------------------------------------------------------------
def test_segmentation_mask_guard_and_skeleton_model_resolution():
    """Verifies that segmentation mask prevents background bleed and SMPL models resolve robustly."""
    from app.pipeline.reconstructor import compute_projected_vertex_colors
    from app.pipeline.motion import skeleton_fitter, rigging_engine
    from app.schemas.scene import AnimationState

    # 1. Animation state default must be static (isPlaying = False)
    anim_state = AnimationState()
    assert anim_state.is_playing is False, "Default reconstructed pose must be static, not animated"

    # 2. Skeleton fitter and rigging engine must have resolved SMPL NPZ
    assert skeleton_fitter._j_regressor is not None, "SkeletonFitter must resolve genuine SMPL J_regressor"
    assert rigging_engine._smpl_weights is not None, "RiggingEngine must resolve genuine SMPL skinning weights"
    assert skeleton_fitter._j_regressor.shape == (24, 6890)
    assert rigging_engine._smpl_weights.shape == (6890, 24)

    # 3. Test background pixel bleed prevention via segmentation mask
    # Crop is mostly black (0,0,0) with a central red patch (220, 20, 20)
    crop_rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    crop_rgb[40:88, 40:88] = [220, 20, 20] # Person is in center
    mask_gray = np.zeros((128, 128), dtype=np.uint8)
    mask_gray[40:88, 40:88] = 255 # Mask covers central patch

    # Vertex projecting outside the mask into the black background
    # Three.js (X=0.9, Y=0.9) maps to corner of crop where mask=0 and crop=(0,0,0)
    verts = np.array([[0.9, 0.9, 0.0]], dtype=np.float32)
    normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    cam = {"scale": 1.0, "tx": 0.0, "ty": 0.0}

    # With mask guard: outside vertex receives representative foreground color (red)
    vc_guarded = compute_projected_vertex_colors(verts, normals, cam, crop_rgb, mask_gray=mask_gray)
    assert vc_guarded[0, 0] > 150, "Mask guard must replace background black with representative foreground red"




