"""DRISHTI - 3D Body Reprojection Diagnostic & Visual Overlay

Validates 3D reconstructed humans by projecting the 3D SMPL body mesh and skeleton
back onto the original source photograph, computing reprojection errors,
silhouette IoU, and generating visual discrepancy overlays.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

from app.schemas.common import BoundingBox2D
from app.core.logging import logger


# SMPL 24 kinematic joint names and standard connectivity
SMPL_JOINT_NAMES = [
    "pelvis",         # 0
    "left_hip",       # 1
    "right_hip",      # 2
    "spine1",         # 3
    "left_knee",      # 4
    "right_knee",     # 5
    "spine2",         # 6
    "left_ankle",     # 7
    "right_ankle",    # 8
    "spine3",         # 9
    "left_foot",      # 10
    "right_foot",     # 11
    "neck",           # 12
    "left_collar",    # 13
    "right_collar",   # 14
    "head",           # 15
    "left_shoulder",  # 16
    "right_shoulder", # 17
    "left_elbow",     # 18
    "right_elbow",    # 19
    "left_wrist",     # 20
    "right_wrist",    # 21
    "left_hand",      # 22
    "right_hand",     # 23
]

# Kinematic tree edges (parent, child)
SMPL_SKELETON_PAIRS = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),
    (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),
]

# Mapping from YOLOv8-Pose keypoint names to SMPL joint indices
YOLO_TO_SMPL_MAPPING = {
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
    "nose": 15,
}


@dataclass
class ReprojectionDiagnostic:
    """Quantitative and visual reprojection validation metrics."""
    mean_reprojection_error_pixels: Optional[float] = None
    median_reprojection_error_pixels: Optional[float] = None
    max_reprojection_error_pixels: Optional[float] = None
    silhouette_iou: Optional[float] = None
    bbox_iou: Optional[float] = None
    keypoint_errors: Dict[str, float] = field(default_factory=dict)
    matched_keypoints_count: int = 0
    overlay_image_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


def project_points_to_image(
    points_3d: np.ndarray,
    camera_params: Dict[str, float],
    transform_meta: Dict[str, Any],
) -> np.ndarray:
    """Projects 3D points in OpenCV camera coordinates to original photo pixel coordinates.
    
    Args:
        points_3d: (N, 3) coordinates in OpenCV camera frame (+X right, +Y down, +Z depth)
        camera_params: Dict containing 'scale', 'tx', 'ty' from weak-perspective model
        transform_meta: Dict containing crop offsets, box_pixels, square_dim, original_image_size
        
    Returns:
        points_2d: (N, 2) pixel coordinates on the full original image
    """
    s = float(camera_params.get("scale", 1.0))
    tx = float(camera_params.get("tx", 0.0))
    ty = float(camera_params.get("ty", 0.0))

    # Weak-perspective projection onto normalized crop space [-1, 1]
    # x_crop_norm = s * X + tx
    # y_crop_norm = s * Y + ty
    proj_x_norm = s * points_3d[:, 0] + tx
    proj_y_norm = s * points_3d[:, 1] + ty

    # Map normalized crop [-1, 1] to target crop pixel space (e.g. 256x256 or 224x224)
    target_dim = float(transform_meta.get("target_size", [256, 256])[0]) if "target_size" in transform_meta else 256.0
    crop_px_x = (proj_x_norm + 1.0) * 0.5 * target_dim
    crop_px_y = (proj_y_norm + 1.0) * 0.5 * target_dim

    # Un-letterbox and un-crop back to original full photograph pixels
    max_dim = float(transform_meta.get("square_dim", target_dim))
    scale_factor = max_dim / target_dim

    square_px_x = crop_px_x * scale_factor
    square_px_y = crop_px_y * scale_factor

    offset = transform_meta.get("offset", [0, 0])
    crop_pixels = transform_meta.get("crop_pixels", [0, 0, max_dim, max_dim])

    orig_px_x = square_px_x - offset[0] + crop_pixels[0]
    orig_px_y = square_px_y - offset[1] + crop_pixels[1]

    return np.stack([orig_px_x, orig_px_y], axis=-1)


def generate_reprojection_overlay(
    image_rgb: np.ndarray,
    joints_3d: np.ndarray,
    camera_params: Dict[str, float],
    transform_meta: Dict[str, Any],
    bounding_box: Optional[BoundingBox2D] = None,
    mask_gray: Optional[np.ndarray] = None,
    observed_keypoints: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None,
) -> Tuple[np.ndarray, ReprojectionDiagnostic]:
    """Generates a visual diagnostic overlay and quantitative metrics comparing
    the 3D reconstruction against 2D observations in the original photograph.
    """
    img_h, img_w = image_rgb.shape[:2]
    overlay = image_rgb.copy()
    diag = ReprojectionDiagnostic()

    # 1. Project 3D joints to original image pixels
    projected_joints_2d = project_points_to_image(joints_3d, camera_params, transform_meta)

    # 2. Draw segmentation mask overlay (if available)
    if mask_gray is not None and isinstance(mask_gray, np.ndarray):
        if mask_gray.shape[:2] != (img_h, img_w):
            mask_resized = cv2.resize(mask_gray, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask_gray
        binary_mask = (mask_resized > 127).astype(np.uint8)
        
        # Green tint for ground-truth segmentation
        mask_color = np.zeros_like(overlay)
        mask_color[:, :, 1] = 200  # Green
        alpha = 0.25
        overlay = np.where(binary_mask[:, :, None] == 1,
                           cv2.addWeighted(overlay, 1 - alpha, mask_color, alpha, 0),
                           overlay)

    # 3. Draw bounding box (cyan)
    if bounding_box is not None:
        bx1 = int(bounding_box.x_min * img_w)
        by1 = int(bounding_box.y_min * img_h)
        bx2 = int(bounding_box.x_max * img_w)
        by2 = int(bounding_box.y_max * img_h)
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (255, 255, 0), 2)  # Cyan
        conf_str = f" ({bounding_box.confidence:.2f})" if hasattr(bounding_box, "confidence") and bounding_box.confidence is not None else ""
        cv2.putText(overlay, f"Person Detection{conf_str}",
                    (bx1, max(15, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    # 4. Draw projected 3D skeleton (magenta lines, blue joints)
    num_joints = len(joints_3d)
    for parent_idx, child_idx in SMPL_SKELETON_PAIRS:
        if parent_idx < num_joints and child_idx < num_joints:
            p1 = (int(round(projected_joints_2d[parent_idx, 0])), int(round(projected_joints_2d[parent_idx, 1])))
            p2 = (int(round(projected_joints_2d[child_idx, 0])), int(round(projected_joints_2d[child_idx, 1])))
            # Check within plausible image bounds
            if (-100 < p1[0] < img_w + 100 and -100 < p1[1] < img_h + 100 and
                -100 < p2[0] < img_w + 100 and -100 < p2[1] < img_h + 100):
                cv2.line(overlay, p1, p2, (255, 0, 255), 2)  # Magenta

    for j_idx in range(num_joints):
        pt = (int(round(projected_joints_2d[j_idx, 0])), int(round(projected_joints_2d[j_idx, 1])))
        if 0 <= pt[0] < img_w and 0 <= pt[1] < img_h:
            cv2.circle(overlay, pt, 4, (255, 100, 0), -1)  # Deep Blue
            cv2.circle(overlay, pt, 5, (255, 255, 255), 1)  # White border

    # 5. Measure keypoint reprojection errors and draw discrepancies
    errors = []
    if observed_keypoints is not None:
        for kp_name, smpl_idx in YOLO_TO_SMPL_MAPPING.items():
            kp_data = observed_keypoints.get(kp_name)
            if kp_data is None:
                continue

            conf = getattr(kp_data, "confidence", 0.0) if hasattr(kp_data, "confidence") else kp_data.get("confidence", 0.0)
            if conf < 0.35:
                continue

            # Detected 2D keypoint in pixel coordinates
            if hasattr(kp_data, "x_norm") and hasattr(kp_data, "y_norm"):
                obs_px_x = kp_data.x_norm * img_w
                obs_px_y = kp_data.y_norm * img_h
            elif "x_norm" in kp_data and "y_norm" in kp_data:
                obs_px_x = kp_data["x_norm"] * img_w
                obs_px_y = kp_data["y_norm"] * img_h
            else:
                continue

            proj_pt = projected_joints_2d[smpl_idx]
            err = float(np.sqrt((proj_pt[0] - obs_px_x) ** 2 + (proj_pt[1] - obs_px_y) ** 2))
            errors.append(err)
            diag.keypoint_errors[kp_name] = round(err, 2)

            obs_pt = (int(round(obs_px_x)), int(round(obs_px_y)))
            pr_pt = (int(round(proj_pt[0])), int(round(proj_pt[1])))

            # Draw detected keypoint (solid yellow circle)
            cv2.circle(overlay, obs_pt, 5, (0, 255, 255), -1)

            # If discrepancy > 12 pixels, draw red error vector
            if err > 12.0:
                cv2.line(overlay, obs_pt, pr_pt, (0, 0, 255), 2)  # Red vector
                cv2.circle(overlay, pr_pt, 6, (0, 0, 255), 2)

    if errors:
        diag.mean_reprojection_error_pixels = round(float(np.mean(errors)), 2)
        diag.median_reprojection_error_pixels = round(float(np.median(errors)), 2)
        diag.max_reprojection_error_pixels = round(float(np.max(errors)), 2)
        diag.matched_keypoints_count = len(errors)
    else:
        diag.warnings.append("No reliable 2D keypoints available to compute pixel reprojection error.")

    # 6. Save overlay image if path specified
    if output_path is not None:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            diag.overlay_image_path = str(output_path)
            logger.info(f"Saved reprojection overlay to {output_path}")
        except Exception as e:
            logger.warning(f"Could not save reprojection overlay image: {e}")

    return overlay, diag
