"""DRISHTI - Motion Subsystem (Phase 07: DRISHTI MOTION)

Provides 3D humanoid skeleton representation, hierarchy validation, 2D pose estimation,
3D skeleton-to-mesh fitting, skeletal rigging with skinning weights, and deterministic
procedural animation generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.logging import logger
from app.pipeline.assembly import compute_2d_iou
from app.pipeline.models import DEFAULT_MODELS_DIR
from app.schemas.common import BoundingBox2D, Transform
from app.schemas.scene import (
    AnimationClipData,
    BoneDefinition,
    CoordinateConvention,
    JointTransformTrack,
    RigData,
    SceneEntity,
    SkeletonData,
    SkeletonJoint,
)

# ---------------------------------------------------------------------------
# Canonical 24-Joint Humanoid Hierarchy (Aligned with SMPL Topology)
# ---------------------------------------------------------------------------

CANONICAL_JOINT_NAMES: List[str] = [
    "pelvis",          # 0 (root)
    "left_hip",        # 1
    "right_hip",       # 2
    "spine1",          # 3
    "left_knee",       # 4
    "right_knee",      # 5
    "spine2",          # 6
    "left_ankle",      # 7
    "right_ankle",     # 8
    "spine3",          # 9 (chest)
    "left_foot",       # 10
    "right_foot",      # 11
    "neck",            # 12
    "left_collar",     # 13
    "right_collar",    # 14
    "head",            # 15
    "left_shoulder",   # 16
    "right_shoulder",  # 17
    "left_elbow",      # 18
    "right_elbow",     # 19
    "left_wrist",      # 20
    "right_wrist",     # 21
    "left_hand",       # 22
    "right_hand",      # 23
]

CANONICAL_JOINT_INDICES: Dict[str, int] = {
    name: idx for idx, name in enumerate(CANONICAL_JOINT_NAMES)
}

CANONICAL_PARENTS: Dict[str, Optional[str]] = {
    "pelvis": None,
    "left_hip": "pelvis",
    "right_hip": "pelvis",
    "spine1": "pelvis",
    "left_knee": "left_hip",
    "right_knee": "right_hip",
    "spine2": "spine1",
    "left_ankle": "left_knee",
    "right_ankle": "right_knee",
    "spine3": "spine2",
    "left_foot": "left_ankle",
    "right_foot": "right_ankle",
    "neck": "spine3",
    "left_collar": "spine3",
    "right_collar": "spine3",
    "head": "neck",
    "left_shoulder": "left_collar",
    "right_shoulder": "right_collar",
    "left_elbow": "left_shoulder",
    "right_elbow": "right_shoulder",
    "left_wrist": "left_elbow",
    "right_wrist": "right_elbow",
    "left_hand": "left_wrist",
    "right_hand": "right_wrist",
}

# Anatomical rest positions for a standard 1.75m human centered at pelvis (0, 0, 0)
CANONICAL_REST_POSITIONS: Dict[str, List[float]] = {
    "pelvis": [0.0, 0.0, 0.0],
    "left_hip": [-0.10, -0.08, 0.0],
    "right_hip": [0.10, -0.08, 0.0],
    "spine1": [0.0, 0.12, -0.01],
    "left_knee": [-0.11, -0.48, -0.01],
    "right_knee": [0.11, -0.48, -0.01],
    "spine2": [0.0, 0.26, -0.01],
    "left_ankle": [-0.11, -0.85, -0.02],
    "right_ankle": [0.11, -0.85, -0.02],
    "spine3": [0.0, 0.38, 0.01],
    "left_foot": [-0.11, -0.92, 0.10],
    "right_foot": [0.11, -0.92, 0.10],
    "neck": [0.0, 0.52, 0.0],
    "left_collar": [-0.08, 0.44, 0.0],
    "right_collar": [0.08, 0.44, 0.0],
    "head": [0.0, 0.68, 0.02],
    "left_shoulder": [-0.20, 0.44, 0.0],
    "right_shoulder": [0.20, 0.44, 0.0],
    "left_elbow": [-0.28, 0.18, 0.0],
    "right_elbow": [0.28, 0.18, 0.0],
    "left_wrist": [-0.34, -0.06, 0.0],
    "right_wrist": [0.34, -0.06, 0.0],
    "left_hand": [-0.38, -0.16, 0.0],
    "right_hand": [0.38, -0.16, 0.0],
}

# Mapping from COCO 17 keypoint indices to canonical joint names
COCO_17_KEYPOINTS: List[str] = [
    "nose",           # 0 -> head
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5 -> left_shoulder
    "right_shoulder", # 6 -> right_shoulder
    "left_elbow",     # 7 -> left_elbow
    "right_elbow",    # 8 -> right_elbow
    "left_wrist",     # 9 -> left_wrist
    "right_wrist",    # 10 -> right_wrist
    "left_hip",       # 11 -> left_hip
    "right_hip",      # 12 -> right_hip
    "left_knee",      # 13 -> left_knee
    "right_knee",     # 14 -> right_knee
    "left_ankle",     # 15 -> left_ankle
    "right_ankle",    # 16 -> right_ankle
]

COCO_TO_CANONICAL: Dict[int, str] = {
    0: "head",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}


# ---------------------------------------------------------------------------
# Skeleton Hierarchy Validation Utility
# ---------------------------------------------------------------------------

def validate_skeleton_hierarchy(skeleton: SkeletonData) -> Tuple[bool, List[str]]:
    """Strictly validates a skeleton's topology, unique joints, cycle freedom, and coordinates."""
    issues: List[str] = []

    if not skeleton.joints:
        return False, ["Skeleton contains no joints."]

    joint_ids = set()
    root_count = 0
    joint_map: Dict[str, SkeletonJoint] = {}

    for j in skeleton.joints:
        if j.id in joint_ids:
            issues.append(f"Duplicate joint identifier detected: '{j.id}'.")
        joint_ids.add(j.id)
        joint_map[j.id] = j

        if j.parent_id is None or j.parent_id == "":
            root_count += 1
            if j.id != skeleton.root_joint_id:
                issues.append(f"Root joint mismatch: defined as '{skeleton.root_joint_id}' but found '{j.id}'.")

        # Check coordinates are finite numbers
        if not np.all(np.isfinite(j.position)):
            issues.append(f"Non-finite position coordinates in joint '{j.id}': {j.position}.")
        if not np.all(np.isfinite(j.rotation)):
            issues.append(f"Non-finite rotation quaternion in joint '{j.id}': {j.rotation}.")

    if root_count == 0:
        issues.append("No root joint found (every joint has a parent).")
    elif root_count > 1:
        issues.append(f"Multiple root joints detected ({root_count}). A valid skeleton must have exactly one root.")

    # Check parent references exist
    for j in skeleton.joints:
        if j.parent_id is not None and j.parent_id != "":
            if j.parent_id not in joint_ids:
                issues.append(f"Joint '{j.id}' references non-existent parent '{j.parent_id}'.")

    # Cycle detection via graph traversal
    visited = set()
    for start_id in joint_ids:
        curr = start_id
        path = []
        while curr is not None and curr in joint_map:
            if curr in path:
                issues.append(f"Circular parent-child relationship detected involving joint '{curr}'.")
                break
            path.append(curr)
            curr = joint_map[curr].parent_id

    # Check required canonical joints
    missing_required = set(CANONICAL_JOINT_NAMES) - joint_ids
    if missing_required:
        issues.append(f"Skeleton is missing required canonical joints: {sorted(list(missing_required))}.")

    is_valid = len(issues) == 0
    return is_valid, issues


# ---------------------------------------------------------------------------
# Pose Estimation Adapters
# ---------------------------------------------------------------------------

@dataclass
class PoseKeypoint2D:
    x_norm: float
    y_norm: float
    confidence: float
    is_visible: bool


@dataclass
class PoseEstimationResult:
    bounding_box: BoundingBox2D
    keypoints: Dict[str, PoseKeypoint2D]
    confidence_score: float
    model_name: str


class IPoseEstimator(ABC):
    """Abstract interface for local 2D human pose estimation."""

    @abstractmethod
    def estimate_poses(
        self,
        image_bgr: np.ndarray,
        entity_boxes: Optional[List[BoundingBox2D]] = None,
    ) -> List[PoseEstimationResult]:
        pass


class MockPoseEstimator(IPoseEstimator):
    """Deterministic mock pose estimator for fast unit testing and offline CI."""

    def __init__(self):
        self.model_name = "MockPoseEstimator-Deterministic"

    def estimate_poses(
        self,
        image_bgr: np.ndarray,
        entity_boxes: Optional[List[BoundingBox2D]] = None,
    ) -> List[PoseEstimationResult]:
        results: List[PoseEstimationResult] = []
        boxes = entity_boxes or [BoundingBox2D(x_min=0.1, y_min=0.05, x_max=0.9, y_max=0.95)]

        for box in boxes:
            cx = (box.x_min + box.x_max) / 2.0
            cy = (box.y_min + box.y_max) / 2.0
            bw = box.x_max - box.x_min
            bh = box.y_max - box.y_min

            kps: Dict[str, PoseKeypoint2D] = {
                "head": PoseKeypoint2D(cx, box.y_min + bh * 0.12, 0.95, True),
                "left_shoulder": PoseKeypoint2D(cx - bw * 0.22, box.y_min + bh * 0.25, 0.92, True),
                "right_shoulder": PoseKeypoint2D(cx + bw * 0.22, box.y_min + bh * 0.25, 0.92, True),
                "left_elbow": PoseKeypoint2D(cx - bw * 0.32, box.y_min + bh * 0.45, 0.88, True),
                "right_elbow": PoseKeypoint2D(cx + bw * 0.32, box.y_min + bh * 0.45, 0.88, True),
                "left_wrist": PoseKeypoint2D(cx - bw * 0.38, box.y_min + bh * 0.65, 0.85, True),
                "right_wrist": PoseKeypoint2D(cx + bw * 0.38, box.y_min + bh * 0.65, 0.85, True),
                "left_hip": PoseKeypoint2D(cx - bw * 0.14, box.y_min + bh * 0.52, 0.90, True),
                "right_hip": PoseKeypoint2D(cx + bw * 0.14, box.y_min + bh * 0.52, 0.90, True),
                "left_knee": PoseKeypoint2D(cx - bw * 0.15, box.y_min + bh * 0.72, 0.89, True),
                "right_knee": PoseKeypoint2D(cx + bw * 0.15, box.y_min + bh * 0.72, 0.89, True),
                "left_ankle": PoseKeypoint2D(cx - bw * 0.15, box.y_max - bh * 0.05, 0.86, True),
                "right_ankle": PoseKeypoint2D(cx + bw * 0.15, box.y_max - bh * 0.05, 0.86, True),
            }

            results.append(
                PoseEstimationResult(
                    bounding_box=box,
                    keypoints=kps,
                    confidence_score=0.91,
                    model_name=self.model_name,
                )
            )

        return results


class YoloPoseEstimator(IPoseEstimator):
    """Ultralytics YOLOv8 Pose estimation adapter for genuine local multi-person inference."""

    def __init__(self, model_path: Path, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model_name = "YOLOv8-Pose"
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from ultralytics import YOLO
            logger.info(f"Loading YOLOv8 Pose model from '{self.model_path}' on device '{self.device}'...")
            self._model = YOLO(str(self.model_path))

    def estimate_poses(
        self,
        image_bgr: np.ndarray,
        entity_boxes: Optional[List[BoundingBox2D]] = None,
    ) -> List[PoseEstimationResult]:
        self._ensure_loaded()
        h, w = image_bgr.shape[:2]
        pred_results = self._model(image_bgr, device=self.device, verbose=False)
        r = pred_results[0]

        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            return []

        poses: List[PoseEstimationResult] = []
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        kpts_xy = r.keypoints.xy.cpu().numpy()  # (N, 17, 2)
        kpts_conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones((len(boxes_xyxy), 17))

        for idx in range(len(boxes_xyxy)):
            b = boxes_xyxy[idx]
            box_obj = BoundingBox2D(
                x_min=float(b[0] / w),
                y_min=float(b[1] / h),
                x_max=float(b[2] / w),
                y_max=float(b[3] / h),
            )

            kps: Dict[str, PoseKeypoint2D] = {}
            for coco_idx, canonical_name in COCO_TO_CANONICAL.items():
                px = float(kpts_xy[idx, coco_idx, 0] / w)
                py = float(kpts_xy[idx, coco_idx, 1] / h)
                conf = float(kpts_conf[idx, coco_idx])
                kps[canonical_name] = PoseKeypoint2D(
                    x_norm=px,
                    y_norm=py,
                    confidence=conf,
                    is_visible=conf > 0.3,
                )

            poses.append(
                PoseEstimationResult(
                    bounding_box=box_obj,
                    keypoints=kps,
                    confidence_score=float(scores[idx]),
                    model_name=self.model_name,
                )
            )

        if entity_boxes is not None:
            if not poses:
                return []

            matched: List[Optional[PoseEstimationResult]] = [None] * len(entity_boxes)
            unassigned_pose_indices = set(range(len(poses)))

            # Phase 1: High-confidence IoU bipartite matching
            iou_candidates = []
            for e_idx, box in enumerate(entity_boxes):
                for p_idx, p in enumerate(poses):
                    iou = compute_2d_iou(box, p.bounding_box)
                    if iou > 0.15:
                        iou_candidates.append((iou, e_idx, p_idx))

            iou_candidates.sort(key=lambda x: x[0], reverse=True)
            for iou, e_idx, p_idx in iou_candidates:
                if matched[e_idx] is None and p_idx in unassigned_pose_indices:
                    matched[e_idx] = poses[p_idx]
                    unassigned_pose_indices.remove(p_idx)

            # Phase 2: Proximity matching by normalized center Euclidean distance
            for e_idx, box in enumerate(entity_boxes):
                if matched[e_idx] is None:
                    bcx = (box.x_min + box.x_max) / 2.0
                    bcy = (box.y_min + box.y_max) / 2.0

                    best_p_idx = -1
                    best_dist = float("inf")
                    pool = unassigned_pose_indices if unassigned_pose_indices else range(len(poses))
                    for p_idx in pool:
                        p = poses[p_idx]
                        pcx = (p.bounding_box.x_min + p.bounding_box.x_max) / 2.0
                        pcy = (p.bounding_box.y_min + p.bounding_box.y_max) / 2.0
                        dist = (bcx - pcx) ** 2 + (bcy - pcy) ** 2
                        if dist < best_dist:
                            best_dist = dist
                            best_p_idx = p_idx

                    if best_p_idx >= 0:
                        matched[e_idx] = poses[best_p_idx]
                        if best_p_idx in unassigned_pose_indices:
                            unassigned_pose_indices.remove(best_p_idx)

            return [m if m is not None else poses[0] for m in matched]

        return poses


# ---------------------------------------------------------------------------
# 3D Skeleton Fitter
# ---------------------------------------------------------------------------

class SkeletonFitter:
    """Builds and aligns a canonical 3D skeleton for a reconstructed person entity."""

    def __init__(self, smpl_npz_path: Optional[Path] = None):
        self.smpl_npz_path = smpl_npz_path
        self._j_regressor = None
        if smpl_npz_path and smpl_npz_path.exists():
            try:
                npz = np.load(smpl_npz_path, allow_pickle=True)
                if "J_regressor" in npz:
                    self._j_regressor = npz["J_regressor"]
            except Exception as e:
                logger.warning(f"Could not load SMPL J_regressor: {e}")

    def build_skeleton(
        self,
        entity: SceneEntity,
        mesh_vertices: Optional[np.ndarray] = None,
        pose_result: Optional[PoseEstimationResult] = None,
    ) -> SkeletonData:
        """Constructs a 24-joint canonical skeleton aligned with the reconstructed mesh."""
        joint_positions: Dict[str, List[float]] = {}
        joint_confidences: Dict[str, float] = {}
        source = "mock_anatomical"

        # Strategy 1: Genuine SMPL J_regressor from 6890 mesh vertices
        if mesh_vertices is not None and len(mesh_vertices) == 6890 and self._j_regressor is not None:
            try:
                # J = J_regressor * V (24, 3)
                j_3d = np.matmul(self._j_regressor, mesh_vertices)
                for idx, name in enumerate(CANONICAL_JOINT_NAMES):
                    joint_positions[name] = [float(j_3d[idx, 0]), float(j_3d[idx, 1]), float(j_3d[idx, 2])]
                    joint_confidences[name] = 0.95
                source = "smpl_regressed"
            except Exception as e:
                logger.warning(f"Error regressing SMPL joints: {e}")

        # Strategy 2: Landmark projection from mesh bounding box
        if not joint_positions:
            height = 1.75
            width = 0.5
            if mesh_vertices is not None and len(mesh_vertices) > 0:
                dims = mesh_vertices.max(axis=0) - mesh_vertices.min(axis=0)
                height = float(dims[1])
                width = float(dims[0])

            scale_y = height / 1.75
            scale_x = width / 0.5
            for name, rest_pos in CANONICAL_REST_POSITIONS.items():
                joint_positions[name] = [
                    float(rest_pos[0] * scale_x),
                    float(rest_pos[1] * scale_y),
                    float(rest_pos[2]),
                ]
                joint_confidences[name] = 0.90
            source = "mock_anatomical"

        # Correlate 2D pose confidence if available
        if pose_result:
            for joint_name, kp in pose_result.keypoints.items():
                if joint_name in joint_confidences:
                    joint_confidences[joint_name] = kp.confidence

        # Assemble SkeletonJoint objects
        joints: List[SkeletonJoint] = []
        for idx, name in enumerate(CANONICAL_JOINT_NAMES):
            parent = CANONICAL_PARENTS[name]
            pos = joint_positions.get(name, CANONICAL_REST_POSITIONS[name])
            conf = joint_confidences.get(name, 0.85)
            joints.append(
                SkeletonJoint(
                    id=name,
                    name=name.replace("_", " ").title(),
                    index=idx,
                    parent_id=parent,
                    position=pos,
                    rotation=[0.0, 0.0, 0.0, 1.0],  # Identity quaternion
                    confidence=conf,
                    is_observed=(name in (pose_result.keypoints if pose_result else {})),
                )
            )

        # Assemble BoneDefinition objects
        bones: List[BoneDefinition] = []
        for j in joints:
            if j.parent_id:
                p_pos = np.array(joint_positions[j.parent_id])
                c_pos = np.array(j.position)
                length = float(np.linalg.norm(c_pos - p_pos))
                bones.append(
                    BoneDefinition(
                        id=f"bone-{j.parent_id}-{j.id}",
                        head_joint_id=j.parent_id,
                        tail_joint_id=j.id,
                        length=length,
                    )
                )

        skeleton = SkeletonData(
            id=f"skel-{entity.id}",
            entity_id=entity.id,
            schema_version="1.0.0",
            root_joint_id="pelvis",
            joints=joints,
            bones=bones,
            coordinate_convention=CoordinateConvention(up="+Y", forward="+Z", right="+X", units="meters"),
            source=source,
            is_aligned=True,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        is_valid, issues = validate_skeleton_hierarchy(skeleton)
        skeleton.validation_status = "valid" if is_valid else "invalid"
        skeleton.validation_errors = issues

        return skeleton


# ---------------------------------------------------------------------------
# Rigging Engine
# ---------------------------------------------------------------------------

def validate_skinning_weights(
    weights: np.ndarray,
    bone_indices: np.ndarray,
    bone_count: int = 24,
) -> Tuple[bool, List[str]]:
    """Validates vertex skinning weights for finiteness, index range, and L1 unit sum."""
    issues = []
    if weights is None or len(weights) == 0:
        return False, ["Skinning weights array is empty."]

    if not np.all(np.isfinite(weights)):
        return False, ["Skinning weights contain NaNs or infinite values."]

    if np.any(bone_indices < 0) or np.any(bone_indices >= bone_count):
        issues.append(f"Bone indices out of valid range [0, {bone_count - 1}].")

    sums = np.sum(weights, axis=1)
    if not np.allclose(sums, 1.0, atol=1e-3):
        max_err = float(np.max(np.abs(sums - 1.0)))
        issues.append(f"Skinning weights do not sum to 1.0 (max deviation: {max_err:.4f}).")

    return len(issues) == 0, issues


class RiggingEngine:
    """Associates 3D reconstructed meshes with articulated skeletons via skinning weights."""

    def __init__(self, smpl_npz_path: Optional[Path] = None):
        self.smpl_npz_path = smpl_npz_path
        self._smpl_weights = None
        if smpl_npz_path and smpl_npz_path.exists():
            try:
                npz = np.load(smpl_npz_path, allow_pickle=True)
                if "weights" in npz:
                    self._smpl_weights = npz["weights"]  # (6890, 24)
            except Exception as e:
                logger.warning(f"Could not load SMPL weights: {e}")

    def rig_mesh(
        self,
        entity: SceneEntity,
        skeleton: SkeletonData,
        mesh_vertices: Optional[np.ndarray] = None,
    ) -> RigData:
        """Computes top-4 bone weights per vertex and validates rig integrity."""
        rig_id = f"rig-{entity.id}"

        if mesh_vertices is None or len(mesh_vertices) == 0:
            return RigData(
                id=rig_id,
                entity_id=entity.id,
                skeleton_id=skeleton.id,
                mesh_asset_id=entity.geometry_ref,
                status="unrigged",
                skinning_method="none",
                bone_count=len(skeleton.joints),
                error_message="No mesh geometry available for rigging.",
            )

        num_verts = len(mesh_vertices)

        # Strategy 1: Pretrained SMPL LBS weights (exact topology: 6890 vertices)
        if num_verts == 6890 and self._smpl_weights is not None:
            raw_w = self._smpl_weights
            top4_idx = np.argsort(raw_w, axis=1)[:, -4:]
            top4_w = np.take_along_axis(raw_w, top4_idx, axis=1)
            norm_w = top4_w / np.maximum(np.sum(top4_w, axis=1, keepdims=True), 1e-6)
            skinning_method = "smpl_lbs_native"
        else:
            # Strategy 2: Proximity heat skinning for mock/arbitrary meshes
            joint_positions = np.array([j.position for j in skeleton.joints])  # (24, 3)
            # Distance from each vertex to each joint
            dists = np.linalg.norm(mesh_vertices[:, None, :] - joint_positions[None, :, :], axis=2)  # (N, 24)
            inv_dists = 1.0 / (dists**2 + 0.005)
            top4_idx = np.argsort(inv_dists, axis=1)[:, -4:]
            top4_w = np.take_along_axis(inv_dists, top4_idx, axis=1)
            norm_w = top4_w / np.sum(top4_w, axis=1, keepdims=True)
            skinning_method = "proximity_heat_weights"

        is_valid, issues = validate_skinning_weights(norm_w, top4_idx, bone_count=len(skeleton.joints))

        if not is_valid:
            return RigData(
                id=rig_id,
                entity_id=entity.id,
                skeleton_id=skeleton.id,
                mesh_asset_id=entity.geometry_ref,
                status="failed",
                skinning_method=skinning_method,
                bone_count=len(skeleton.joints),
                weights_normalized=False,
                error_message="; ".join(issues),
                rigged_at=datetime.now(timezone.utc).isoformat(),
            )

        return RigData(
            id=rig_id,
            entity_id=entity.id,
            skeleton_id=skeleton.id,
            mesh_asset_id=entity.geometry_ref,
            status="rigged",
            skinning_method=skinning_method,
            bone_count=len(skeleton.joints),
            max_influences_per_vertex=4,
            weights_normalized=True,
            rigged_at=datetime.now(timezone.utc).isoformat(),
        )


# ---------------------------------------------------------------------------
# Procedural Animation Engine
# ---------------------------------------------------------------------------

def euler_to_quaternion(rx: float, ry: float, rz: float) -> List[float]:
    """Converts Euler rotation angles (in radians, XYZ order) to unit quaternion [qx, qy, qz, qw]."""
    cx = math.cos(rx * 0.5)
    sx = math.sin(rx * 0.5)
    cy = math.cos(ry * 0.5)
    sy = math.sin(ry * 0.5)
    cz = math.cos(rz * 0.5)
    sz = math.sin(rz * 0.5)
    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz
    return [float(qx), float(qy), float(qz), float(qw)]


class ProceduralAnimationEngine:
    """Generates deterministic humanoid test animation clips for valid articulated rigs."""

    @staticmethod
    def generate_clips_for_entity(
        entity_id: str,
        skeleton_id: str,
        is_seated: bool = False,
    ) -> List[AnimationClipData]:
        """Generates deterministic humanoid test animation clips for valid articulated rigs.
        
        When is_seated is True, generates posture-aware seated clips (seated_idle,
        seated_weight_shift, arm_wave) preserving seated hip and knee flexion.
        When is_seated is False, generates standing clips (idle_breathing, torso_twist,
        arm_wave, gentle_walk).
        """
        fps = 30.0

        if is_seated:
            # Clip 1: Seated Idle Breathing (3.5 seconds)
            # Subtle chest elevation and head micro-tilt; hips and knees remain fixed in seated pose
            seated_idle_duration = 3.5
            num_frames = int(seated_idle_duration * fps)
            timestamps = [round(i / fps, 4) for i in range(num_frames)]

            spine3_quats = []
            head_quats = []
            l_sh_quats = []
            r_sh_quats = []
            for t in timestamps:
                phase = (t / seated_idle_duration) * 2.0 * math.pi
                pitch = 0.025 * math.sin(phase)
                spine3_quats.append(euler_to_quaternion(pitch, 0.0, 0.0))
                head_quats.append(euler_to_quaternion(0.01 * math.sin(phase * 0.7), 0.012 * math.cos(phase * 0.5), 0.0))
                roll = 0.015 * math.sin(phase)
                l_sh_quats.append(euler_to_quaternion(0.0, 0.0, -roll))
                r_sh_quats.append(euler_to_quaternion(0.0, 0.0, roll))

            clip_seated_idle = AnimationClipData(
                id=f"clip-seated-idle-{entity_id}",
                name="seated_idle",
                target_entity_id=entity_id,
                target_skeleton_id=skeleton_id,
                duration_seconds=seated_idle_duration,
                is_looping=True,
                fps=fps,
                provenance="procedural",
                tracks=[
                    JointTransformTrack(joint_id="spine3", timestamps=timestamps, rotations=spine3_quats),
                    JointTransformTrack(joint_id="head", timestamps=timestamps, rotations=head_quats),
                    JointTransformTrack(joint_id="left_shoulder", timestamps=timestamps, rotations=l_sh_quats),
                    JointTransformTrack(joint_id="right_shoulder", timestamps=timestamps, rotations=r_sh_quats),
                ],
                created_at=datetime.now(timezone.utc).isoformat(),
            )

            # Clip 2: Seated Weight Shift (4.5 seconds)
            # Subtle upper-body weight shift, torso lean and gentle settling; maintains seated lower body
            shift_duration = 4.5
            num_frames_shift = int(shift_duration * fps)
            shift_timestamps = [round(i / fps, 4) for i in range(num_frames_shift)]

            spine2_shift = []
            head_shift = []
            for t in shift_timestamps:
                phase = (t / shift_duration) * 2.0 * math.pi
                roll = 0.035 * math.sin(phase)
                yaw = 0.025 * math.sin(phase * 0.5)
                spine2_shift.append(euler_to_quaternion(0.0, yaw, roll))
                head_shift.append(euler_to_quaternion(0.0, -yaw * 0.6, -roll * 0.5))

            clip_seated_shift = AnimationClipData(
                id=f"clip-seated-shift-{entity_id}",
                name="seated_weight_shift",
                target_entity_id=entity_id,
                target_skeleton_id=skeleton_id,
                duration_seconds=shift_duration,
                is_looping=True,
                fps=fps,
                provenance="procedural",
                tracks=[
                    JointTransformTrack(joint_id="spine2", timestamps=shift_timestamps, rotations=spine2_shift),
                    JointTransformTrack(joint_id="head", timestamps=shift_timestamps, rotations=head_shift),
                ],
                created_at=datetime.now(timezone.utc).isoformat(),
            )

            # Clip 3: Seated Arm Wave / Gesture (2.5 seconds)
            wave_duration = 2.5
            num_frames_wave = int(wave_duration * fps)
            wave_timestamps = [round(i / fps, 4) for i in range(num_frames_wave)]

            r_sh_wave = []
            r_el_wave = []
            r_wr_wave = []
            for t in wave_timestamps:
                phase = (t / wave_duration) * 2.0 * math.pi
                r_sh_wave.append(euler_to_quaternion(0.3, 0.0, 0.8))
                r_el_wave.append(euler_to_quaternion(0.0, 0.0, 0.6))
                osc = 0.35 * math.sin(phase * 2.0)
                r_wr_wave.append(euler_to_quaternion(0.0, 0.0, osc))

            clip_seated_wave = AnimationClipData(
                id=f"clip-seated-wave-{entity_id}",
                name="arm_wave",
                target_entity_id=entity_id,
                target_skeleton_id=skeleton_id,
                duration_seconds=wave_duration,
                is_looping=True,
                fps=fps,
                provenance="procedural",
                tracks=[
                    JointTransformTrack(joint_id="right_shoulder", timestamps=wave_timestamps, rotations=r_sh_wave),
                    JointTransformTrack(joint_id="right_elbow", timestamps=wave_timestamps, rotations=r_el_wave),
                    JointTransformTrack(joint_id="right_wrist", timestamps=wave_timestamps, rotations=r_wr_wave),
                ],
                created_at=datetime.now(timezone.utc).isoformat(),
            )

            return [clip_seated_idle, clip_seated_shift, clip_seated_wave]

        # Standing Humanoid Clips (Default)
        # Clip 1: Idle Breathing (3.0 seconds)
        idle_duration = 3.0
        num_frames = int(idle_duration * fps)
        timestamps = [round(i / fps, 4) for i in range(num_frames)]

        spine3_quats = []
        l_sh_quats = []
        r_sh_quats = []
        for t in timestamps:
            phase = (t / idle_duration) * 2.0 * math.pi
            # Subtle chest elevation (pitch)
            pitch = 0.035 * math.sin(phase)
            spine3_quats.append(euler_to_quaternion(pitch, 0.0, 0.0))
            # Subtle lateral shoulder sway (roll)
            roll = 0.02 * math.sin(phase)
            l_sh_quats.append(euler_to_quaternion(0.0, 0.0, -roll))
            r_sh_quats.append(euler_to_quaternion(0.0, 0.0, roll))

        clip_idle = AnimationClipData(
            id=f"clip-idle-{entity_id}",
            name="idle_breathing",
            target_entity_id=entity_id,
            target_skeleton_id=skeleton_id,
            duration_seconds=idle_duration,
            is_looping=True,
            fps=fps,
            provenance="procedural",
            tracks=[
                JointTransformTrack(joint_id="spine3", timestamps=timestamps, rotations=spine3_quats),
                JointTransformTrack(joint_id="left_shoulder", timestamps=timestamps, rotations=l_sh_quats),
                JointTransformTrack(joint_id="right_shoulder", timestamps=timestamps, rotations=r_sh_quats),
            ],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Clip 2: Torso Twist (4.0 seconds)
        twist_duration = 4.0
        num_frames_twist = int(twist_duration * fps)
        twist_timestamps = [round(i / fps, 4) for i in range(num_frames_twist)]

        spine2_quats = []
        head_quats = []
        for t in twist_timestamps:
            phase = (t / twist_duration) * 2.0 * math.pi
            yaw = 0.18 * math.sin(phase)
            spine2_quats.append(euler_to_quaternion(0.0, yaw, 0.0))
            head_quats.append(euler_to_quaternion(0.0, -yaw * 0.5, 0.0))

        clip_twist = AnimationClipData(
            id=f"clip-twist-{entity_id}",
            name="torso_twist",
            target_entity_id=entity_id,
            target_skeleton_id=skeleton_id,
            duration_seconds=twist_duration,
            is_looping=True,
            fps=fps,
            provenance="procedural",
            tracks=[
                JointTransformTrack(joint_id="spine2", timestamps=twist_timestamps, rotations=spine2_quats),
                JointTransformTrack(joint_id="head", timestamps=twist_timestamps, rotations=head_quats),
            ],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Clip 3: Arm Wave (2.5 seconds)
        wave_duration = 2.5
        num_frames_wave = int(wave_duration * fps)
        wave_timestamps = [round(i / fps, 4) for i in range(num_frames_wave)]

        r_sh_wave = []
        r_el_wave = []
        r_wr_wave = []
        for t in wave_timestamps:
            phase = (t / wave_duration) * 2.0 * math.pi
            # Right arm raised (roll ~0.8 rad, pitch ~0.3 rad)
            r_sh_wave.append(euler_to_quaternion(0.3, 0.0, 0.8))
            r_el_wave.append(euler_to_quaternion(0.0, 0.0, 0.6))
            # Hand oscillating side to side
            osc = 0.35 * math.sin(phase * 2.0)
            r_wr_wave.append(euler_to_quaternion(0.0, 0.0, osc))

        clip_wave = AnimationClipData(
            id=f"clip-wave-{entity_id}",
            name="arm_wave",
            target_entity_id=entity_id,
            target_skeleton_id=skeleton_id,
            duration_seconds=wave_duration,
            is_looping=True,
            fps=fps,
            provenance="procedural",
            tracks=[
                JointTransformTrack(joint_id="right_shoulder", timestamps=wave_timestamps, rotations=r_sh_wave),
                JointTransformTrack(joint_id="right_elbow", timestamps=wave_timestamps, rotations=r_el_wave),
                JointTransformTrack(joint_id="right_wrist", timestamps=wave_timestamps, rotations=r_wr_wave),
            ],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Clip 4: Gentle Walk Cycle (2.0 seconds)
        walk_duration = 2.0
        num_frames_walk = int(walk_duration * fps)
        walk_timestamps = [round(i / fps, 4) for i in range(num_frames_walk)]

        l_hip_walk = []
        r_hip_walk = []
        l_knee_walk = []
        r_knee_walk = []
        l_arm_walk = []
        r_arm_walk = []
        for t in walk_timestamps:
            phase = (t / walk_duration) * 2.0 * math.pi
            hip_pitch = 0.28 * math.sin(phase)
            l_hip_walk.append(euler_to_quaternion(hip_pitch, 0.0, 0.0))
            r_hip_walk.append(euler_to_quaternion(-hip_pitch, 0.0, 0.0))

            knee_flex_l = max(0.0, 0.35 * math.sin(phase))
            knee_flex_r = max(0.0, 0.35 * math.sin(phase + math.pi))
            l_knee_walk.append(euler_to_quaternion(-knee_flex_l, 0.0, 0.0))
            r_knee_walk.append(euler_to_quaternion(-knee_flex_r, 0.0, 0.0))

            # Counter arm swing
            l_arm_walk.append(euler_to_quaternion(-hip_pitch * 0.8, 0.0, 0.0))
            r_arm_walk.append(euler_to_quaternion(hip_pitch * 0.8, 0.0, 0.0))

        clip_walk = AnimationClipData(
            id=f"clip-walk-{entity_id}",
            name="gentle_walk",
            target_entity_id=entity_id,
            target_skeleton_id=skeleton_id,
            duration_seconds=walk_duration,
            is_looping=True,
            fps=fps,
            provenance="procedural",
            tracks=[
                JointTransformTrack(joint_id="left_hip", timestamps=walk_timestamps, rotations=l_hip_walk),
                JointTransformTrack(joint_id="right_hip", timestamps=walk_timestamps, rotations=r_hip_walk),
                JointTransformTrack(joint_id="left_knee", timestamps=walk_timestamps, rotations=l_knee_walk),
                JointTransformTrack(joint_id="right_knee", timestamps=walk_timestamps, rotations=r_knee_walk),
                JointTransformTrack(joint_id="left_shoulder", timestamps=walk_timestamps, rotations=l_arm_walk),
                JointTransformTrack(joint_id="right_shoulder", timestamps=walk_timestamps, rotations=r_arm_walk),
            ],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        return [clip_idle, clip_twist, clip_wave, clip_walk]


# Module singletons
_smpl_model_path = (DEFAULT_MODELS_DIR / "human_reconstruction" / "SMPL_N_model.npz").resolve()
skeleton_fitter = SkeletonFitter(_smpl_model_path if _smpl_model_path.exists() else None)
rigging_engine = RiggingEngine(_smpl_model_path if _smpl_model_path.exists() else None)
procedural_animation_engine = ProceduralAnimationEngine()

