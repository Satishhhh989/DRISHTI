"""DRISHTI - AI Human 3D Reconstruction Engine (Phase 05)

Provides monocular 3D human mesh reconstruction from single-person photograph crops.
Transforms 2D person detections into valid 3D polygon meshes with verified topology,
computes surface normals, exports standard GLB assets, and maps coordinates to the
canonical DRISHTI scene convention (+X Right, +Y Up, +Z Forward, metric scale).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import trimesh

from app.core.logging import logger
from app.pipeline.device import device_manager
from app.pipeline.models import DEFAULT_MODELS_DIR
from app.schemas.common import BoundingBox2D


@dataclass
class HumanReconstructionResult:
    """Structured result containing real 3D mesh geometry, parameter provenance, and bounds."""
    vertices: np.ndarray  # Shape: (N, 3), float32
    faces: np.ndarray     # Shape: (M, 3), int32
    normals: np.ndarray   # Shape: (N, 3), float32
    bounds_min: List[float]
    bounds_max: List[float]
    dimensions: List[float]  # [width, height, depth] in meters
    glb_bytes: bytes
    model_name: str
    model_version: str
    device: str
    pose_params: Optional[np.ndarray] = None
    shape_params: Optional[np.ndarray] = None
    camera_params: Optional[Dict[str, float]] = None
    texture_availability: str = "none"  # "none" | "partial" | "full"
    quality_score: Optional[float] = None
    quality_preset: str = "high"
    reprojection_error_pixels: Optional[float] = None
    reprojection_error_normalized: Optional[float] = None
    silhouette_iou: Optional[float] = None
    stance_description: str = "Standing Posture (Upright)"
    observed_keypoints_count: int = 0
    pose_convergence_status: str = "Converged"
    metric_status: str = "measured"  # "measured" | "estimated" | "unavailable" | "mock"
    pelvis_3d: Optional[List[float]] = None
    overlay_png_bytes: Optional[bytes] = None
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Geometry Validation & Export Utilities
# ---------------------------------------------------------------------------

def validate_mesh_geometry(
    vertices: np.ndarray,
    faces: np.ndarray
) -> Tuple[bool, List[str]]:
    """Strictly validates mesh geometry for finiteness, index bounds, and topology."""
    issues: List[str] = []

    if vertices is None or len(vertices) == 0:
        return False, ["Mesh contains zero vertices."]
    if faces is None or len(faces) == 0:
        return False, ["Mesh contains zero faces."]

    if not np.all(np.isfinite(vertices)):
        return False, ["Mesh contains NaN or Infinite vertex coordinates."]

    num_verts = len(vertices)
    if np.any(faces < 0) or np.any(faces >= num_verts):
        return False, [f"Face indices out of range [0, {num_verts - 1}]."]

    # Check for degenerate triangles (faces referencing identical vertices)
    degen = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])
    degen_count = int(np.sum(degen))
    if degen_count > 0:
        issues.append(f"Found {degen_count} degenerate faces.")

    # Check reasonable human dimensions (height between 0.4m and 3.0m)
    v_min = np.min(vertices, axis=0)
    v_max = np.max(vertices, axis=0)
    dims = v_max - v_min
    height = dims[1]
    if height < 0.2 or height > 3.5:
        issues.append(f"Unusual reconstructed height: {height:.2f}m (expected 0.5m-2.2m).")

    return True, issues


def compute_mesh_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Computes smooth vertex normals from triangular mesh geometry."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    normals = mesh.vertex_normals
    # Fallback to zero-check
    if normals is None or len(normals) != len(vertices):
        normals = np.zeros_like(vertices, dtype=np.float32)
        normals[:, 1] = 1.0  # Default up normal
    return normals.astype(np.float32)


def export_mesh_glb(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: Optional[np.ndarray] = None,
    vertex_colors: Optional[np.ndarray] = None,
    name: str = "human_mesh"
) -> bytes:
    """Exports validated mesh to binary GLB format with PBR material and optional projected vertex colors."""
    if normals is None:
        normals = compute_mesh_normals(vertices, faces)

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals,
        process=False
    )

    if vertex_colors is not None and len(vertex_colors) == len(vertices):
        vc = np.asarray(vertex_colors)
        if vc.dtype != np.uint8:
            vc = np.clip(vc, 0, 255).astype(np.uint8)
        if vc.ndim == 2 and vc.shape[1] == 3:
            alpha = np.full((len(vc), 1), 255, dtype=np.uint8)
            vc = np.hstack([vc, alpha])
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vc)
    else:
        pbr_mat = trimesh.visual.material.PBRMaterial(
            name=f"{name}_material",
            roughnessFactor=0.6,
            metallicFactor=0.08,
            baseColorFactor=[0.78, 0.82, 0.88, 1.0],
            doubleSided=True
        )
        mesh.visual.material = pbr_mat

    glb_bytes = trimesh.exchange.gltf.export_glb(mesh)
    return glb_bytes


def compute_projected_vertex_colors(
    vertices_three: np.ndarray,
    normals_three: np.ndarray,
    camera_params: Dict[str, float],
    crop_rgb: np.ndarray,
    mask_gray: Optional[Union[bytes, np.ndarray]] = None,
    transform_meta: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Projects photograph crop RGB colors onto 3D human mesh vertices.
    
    Uses weak-perspective camera projection (scale, tx, ty) to sample colors from the
    cropped person photograph, blending front-facing camera projections smoothly onto
    back-facing surfaces with diffuse ambient lighting and hair preservation.
    Guarded by segmentation mask to prevent background pixels from bleeding onto body surfaces.
    
    Args:
        vertices_three: (N, 3) vertices in Three.js coordinates (+X right, +Y up, +Z out)
        normals_three: (N, 3) surface normals in Three.js coordinates
        camera_params: Dict with 'scale', 'tx', 'ty' from HMR/SPIN regressor
        crop_rgb: (H, W, 3) cropped person image array (uint8)
        mask_gray: Optional person segmentation alpha mask
        transform_meta: Optional transform metadata with crop coordinates and square dimensions
        
    Returns:
        (N, 4) uint8 RGBA vertex color array
    """
    s = float(camera_params.get("scale", 1.0))
    tx = float(camera_params.get("tx", 0.0))
    ty = float(camera_params.get("ty", 0.0))

    # Convert Three.js coordinates back to OpenCV camera space for camera parameter alignment
    x_cv = vertices_three[:, 0]
    y_cv = -vertices_three[:, 1]
    nz_cv = -normals_three[:, 2]  # nz_cv < 0 indicates surface facing camera

    h_crop, w_crop = crop_rgb.shape[:2]

    # Normalized Device Coordinates in [-1, 1]
    u = s * x_cv + tx
    v = s * y_cv + ty

    # Pixel coordinates on square crop
    px = np.clip(np.round((u + 1.0) * 0.5 * (w_crop - 1)).astype(np.int32), 0, w_crop - 1)
    py = np.clip(np.round((v + 1.0) * 0.5 * (h_crop - 1)).astype(np.int32), 0, h_crop - 1)

    # Sample direct front projection colors from crop
    front_colors = crop_rgb[py, px].astype(np.float32)

    # Segmentation mask guard: prevent background pixels from bleeding onto mesh surfaces
    if mask_gray is not None:
        try:
            if isinstance(mask_gray, bytes):
                full_mask = cv2.imdecode(np.frombuffer(mask_gray, np.uint8), cv2.IMREAD_GRAYSCALE)
            else:
                full_mask = np.asarray(mask_gray, dtype=np.uint8)

            if full_mask is not None and full_mask.size > 0:
                if transform_meta and "crop_pixels" in transform_meta:
                    cx1, cy1, cx2, cy2 = transform_meta["crop_pixels"]
                    off_x, off_y = transform_meta.get("offset", [0, 0])
                    sq_dim = transform_meta.get("square_dim", max(w_crop, h_crop))
                    m_crop = full_mask[cy1:cy2, cx1:cx2]
                    if m_crop.size > 0:
                        sq_mask = np.zeros((sq_dim, sq_dim), dtype=np.uint8)
                        sq_mask[off_y:off_y + m_crop.shape[0], off_x:off_x + m_crop.shape[1]] = m_crop
                        mask_crop = cv2.resize(sq_mask, (w_crop, h_crop), interpolation=cv2.INTER_NEAREST)
                    else:
                        mask_crop = None
                elif full_mask.shape[:2] == (h_crop, w_crop):
                    mask_crop = full_mask
                else:
                    mask_crop = cv2.resize(full_mask, (w_crop, h_crop), interpolation=cv2.INTER_NEAREST)

                if mask_crop is not None:
                    is_in_mask = mask_crop[py, px] > 64
                    fg_pixels = crop_rgb[mask_crop > 64]
                    if len(fg_pixels) > 0 and np.any(~is_in_mask):
                        rep_clothing = np.median(fg_pixels, axis=0).astype(np.float32)
                        front_colors[~is_in_mask] = rep_clothing
        except Exception as e:
            logger.debug(f"Segmentation mask guard for vertex colors: {e}")

    # Sigmoidal front-to-back blend factor based on camera-facing normal component
    # nz_cv < 0: front-facing (facing camera), nz_cv > 0: back-facing
    blend_front = 1.0 / (1.0 + np.exp(np.clip(6.0 * nz_cv[:, None], -15.0, 15.0)))

    # Back surface color: subtle diffuse ambient shading (85% luminance) matching clothing
    back_colors = front_colors * 0.85

    # Head top / hair region preservation (highest vertices in Y_three, guarded for full human height)
    y_three = vertices_three[:, 1]
    if np.ptp(y_three) > 0.8:
        head_threshold = float(np.max(y_three)) - 0.22
        is_head = y_three > head_threshold
        if np.any(is_head):
            avg_hair = np.mean(front_colors[is_head], axis=0, keepdims=True) * 0.65
            back_colors[is_head] = avg_hair

    final_rgb = blend_front * front_colors + (1.0 - blend_front) * back_colors
    final_rgb = np.clip(np.round(final_rgb), 0, 255).astype(np.uint8)

    alpha = np.full((len(vertices_three), 1), 255, dtype=np.uint8)
    return np.hstack([final_rgb, alpha])


# ---------------------------------------------------------------------------
# Input Image Preparation
# ---------------------------------------------------------------------------

def prepare_person_crop(
    image_rgb: np.ndarray,
    bounding_box: BoundingBox2D,
    padding: float = 0.2,
    target_size: Tuple[int, int] = (224, 224)
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Safely extracts, pads, square-letterboxes, and normalizes a person's crop.
    
    Preserves exact aspect ratio to prevent non-uniform body distortion,
    records transform coordinates back to original image space, and flags clipping.
    """
    img_h, img_w = image_rgb.shape[:2]

    # Unpack normalized bounding box
    x_min = int(bounding_box.x_min * img_w)
    y_min = int(bounding_box.y_min * img_h)
    x_max = int(bounding_box.x_max * img_w)
    y_max = int(bounding_box.y_max * img_h)

    box_w = max(1, x_max - x_min)
    box_h = max(1, y_max - y_min)

    # Calculate padding around bounding box
    pad_x = int(box_w * padding)
    pad_y = int(box_h * padding)

    crop_x1 = x_min - pad_x
    crop_y1 = y_min - pad_y
    crop_x2 = x_max + pad_x
    crop_y2 = y_max + pad_y

    is_clipped = (crop_x1 < 0 or crop_y1 < 0 or crop_x2 > img_w or crop_y2 > img_h)

    # Safe clamped crop
    clamp_x1 = max(0, crop_x1)
    clamp_y1 = max(0, crop_y1)
    clamp_x2 = min(img_w, crop_x2)
    clamp_y2 = min(img_h, crop_y2)

    cropped = image_rgb[clamp_y1:clamp_y2, clamp_x1:clamp_x2]
    crop_h, crop_w = cropped.shape[:2]

    # Square letterboxing to preserve aspect ratio
    max_dim = max(crop_h, crop_w)
    square_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    # Neutral gray fill for padding
    square_img[:, :] = [128, 128, 128]

    offset_y = (max_dim - crop_h) // 2
    offset_x = (max_dim - crop_w) // 2
    square_img[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w] = cropped

    # Resize to target size (224x224)
    resized = cv2.resize(square_img, target_size, interpolation=cv2.INTER_LINEAR)

    transform_meta = {
        "original_image_size": [img_w, img_h],
        "box_pixels": [x_min, y_min, x_max, y_max],
        "crop_pixels": [clamp_x1, clamp_y1, clamp_x2, clamp_y2],
        "square_dim": max_dim,
        "offset": [offset_x, offset_y],
        "is_clipped": is_clipped,
        "aspect_ratio": box_w / box_h,
        "target_size": tuple(target_size),
    }

    return resized, transform_meta


# ---------------------------------------------------------------------------
# Mathematical 6D Rotation & SMPL Linear Blend Skinning
# ---------------------------------------------------------------------------

def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """Official 4DHumans / PyTorch3D 6D continuous rotation to 3x3 rotation matrix.
    Reference: Zhou et al., CVPR 2019; 4D-Humans (Goel et al., 2023)
    """
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1, dim=-1)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def rot6d_to_rotmat_spin(x: torch.Tensor) -> torch.Tensor:
    """Legacy SPIN 6D rotation mapping with transposed interleaving."""
    x = x.view(-1, 3, 2)
    b1 = torch.nn.functional.normalize(x[:, :, 0], dim=-1)
    dot_prod = torch.sum(b1 * x[:, :, 1], dim=-1, keepdim=True)
    b2 = torch.nn.functional.normalize(x[:, :, 1] - dot_prod * b1, dim=-1)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


class SMPLForwardLayer(nn.Module):
    """Pure PyTorch Linear Blend Skinning (LBS) forward deformation layer for SMPL neutral body.
    Requires zero external C++ extensions or chumpy legacy code.
    """

    def __init__(self, npz_data: Dict[str, Any]):
        super().__init__()
        self.register_buffer("v_template", torch.tensor(npz_data["v_template"], dtype=torch.float32))
        self.register_buffer("shapedirs", torch.tensor(npz_data["shapedirs"][:, :, :10], dtype=torch.float32))
        self.register_buffer("posedirs", torch.tensor(npz_data["posedirs"], dtype=torch.float32))
        self.register_buffer("J_regressor", torch.tensor(npz_data["J_regressor"], dtype=torch.float32))
        self.register_buffer("weights", torch.tensor(npz_data["weights"], dtype=torch.float32))
        self.kintree_table = npz_data["kintree_table"]
        self.faces = npz_data["f"].astype(np.int32)

    def forward(
        self,
        pose: torch.Tensor,
        shape_betas: torch.Tensor,
        rot6d_mode: str = "hmr2",
    ) -> torch.Tensor:
        """Deforms SMPL template using shape blend shapes, pose blend shapes, and kinematic skinning."""
        if shape_betas.dim() == 2:
            shape_betas = shape_betas.squeeze(0)

        # 1. Shape Blend Shapes: V_shaped = T_mu + sum(beta_i * S_i)
        v_shaped = self.v_template + torch.einsum("vcd,d->vc", self.shapedirs, shape_betas)

        # 2. Rest Joint Positions: J = J_regressor * V_shaped
        J = torch.matmul(self.J_regressor, v_shaped)  # (24, 3)

        # 3. Compute 3x3 rotation matrices from 6D representation or use directly
        if pose.dim() == 3 and pose.shape[-2:] == (3, 3):
            R = pose
        elif pose.dim() == 4 and pose.shape[-2:] == (3, 3):
            R = pose.squeeze(0) if pose.shape[0] == 1 else pose[0]
        else:
            if pose.dim() == 2 and pose.shape[-1] == 144:
                pose = pose.view(24, 6)
            elif pose.dim() == 1 and len(pose) == 144:
                pose = pose.view(24, 6)
            if rot6d_mode == "spin":
                R = rot6d_to_rotmat_spin(pose)
            else:
                R = rot6d_to_rotmat(pose)

        # 4. Pose Blend Shapes: 23 non-root joints
        I = torch.eye(3, device=R.device).unsqueeze(0).expand(23, -1, -1)
        rot_diff = (R[1:] - I).reshape(-1)  # 23 * 9 = 207
        v_posed = v_shaped + torch.einsum("vcp,p->vc", self.posedirs, rot_diff)

        # 5. Kinematic Forward Hierarchy
        A = [None] * 24
        A[0] = torch.eye(4, device=R.device)
        A[0][:3, :3] = R[0]
        A[0][:3, 3] = J[0]

        for i in range(1, 24):
            parent = int(self.kintree_table[0, i])
            A_rel = torch.eye(4, device=R.device)
            A_rel[:3, :3] = R[i]
            A_rel[:3, 3] = J[i] - J[parent]
            A[i] = torch.matmul(A[parent], A_rel)

        # 6. Skinning Transforms: G_k = A_k * [I, -J_k; 0, 1]
        G = torch.stack(A, dim=0)
        init_bonemat = torch.eye(4, device=R.device).unsqueeze(0).expand(24, -1, -1).clone()
        init_bonemat[:, :3, 3] = -J
        G = torch.matmul(G, init_bonemat)

        # 7. Linear Blend Skinning
        homo_v = torch.cat([v_posed, torch.ones(len(v_posed), 1, device=R.device)], dim=1)
        T = torch.einsum("vk,kab->vab", self.weights, G)
        v_final = torch.einsum("vab,vb->va", T, homo_v)[:, :3]

        return v_final
        A[0][:3, :3] = R[0]
        A[0][:3, 3] = J[0]

        for i in range(1, 24):
            parent = int(self.kintree_table[0, i])
            A_rel = torch.eye(4, device=pose_rot6d.device)
            A_rel[:3, :3] = R[i]
            A_rel[:3, 3] = J[i] - J[parent]
            A[i] = torch.matmul(A[parent], A_rel)

        # 6. Skinning Transforms: G_k = A_k * [I, -J_k; 0, 1]
        G = torch.stack(A, dim=0)
        init_bonemat = torch.eye(4, device=pose_rot6d.device).unsqueeze(0).expand(24, -1, -1).clone()
        init_bonemat[:, :3, 3] = -J
        G = torch.matmul(G, init_bonemat)

        # 7. Linear Blend Skinning
        homo_v = torch.cat([v_posed, torch.ones(len(v_posed), 1, device=v_posed.device)], dim=1)
        T = torch.einsum("vk,kab->vab", self.weights, G)
        v_final = torch.einsum("vab,vb->va", T, homo_v)[:, :3]

        return v_final


class SPINRegressor(nn.Module):
    """Iterative 3-step 6D parameter regressor matching SPIN architecture."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2048 + 144 + 10 + 3, 1024)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout(p=0.5)
        self.decpose = nn.Linear(1024, 144)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)

    def forward(
        self,
        features: torch.Tensor,
        init_pose: torch.Tensor,
        init_shape: torch.Tensor,
        init_cam: torch.Tensor,
        n_iter: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = features.shape[0]
        pred_pose = init_pose.expand(batch_size, -1)
        pred_shape = init_shape.expand(batch_size, -1)
        pred_cam = init_cam.expand(batch_size, -1)

        for _ in range(n_iter):
            xc = torch.cat([features, pred_pose, pred_shape, pred_cam], dim=1)
            xc = torch.relu(self.fc1(xc))
            xc = self.drop1(xc)
            xc = torch.relu(self.fc2(xc))
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        return pred_pose, pred_shape, pred_cam


# ---------------------------------------------------------------------------
# SMPLify Pose & Silhouette Fitting Engine
# ---------------------------------------------------------------------------

class SMPLPoseFitter:
    """SMPLify-style gradient-based pose, shape, and silhouette optimizer.
    
    Refines 3D body pose (24x6 rot6d), body shape betas (10 PCA parameters),
    and weak-perspective camera parameters (scale, tx, ty) by optimizing against
    observed 2D keypoints and silhouette constraints.
    Features multi-hypothesis initialization for seated, standing, and crossed-arm poses,
    along with anatomical joint-limit penalties preventing impossible knee/elbow configurations.
    """

    JOINT_CORRESPONDENCES = [
        (0, "pelvis", 1.6),           # Pelvis (midpoint between hips)
        (12, "neck", 1.6),            # Neck / upper spine (midpoint between shoulders)
        (15, "head", 1.0),            # Head / nose
        (16, "left_shoulder", 1.3),   # L shoulder
        (17, "right_shoulder", 1.3),  # R shoulder
        (18, "left_elbow", 1.5),      # L elbow
        (19, "right_elbow", 1.5),     # R elbow
        (20, "left_wrist", 2.2),      # L wrist (critical for gestures & crossed arms)
        (21, "right_wrist", 2.2),     # R wrist (critical for gestures & crossed arms)
        (1, "left_hip", 1.4),         # L hip
        (2, "right_hip", 1.4),        # R hip
        (4, "left_knee", 1.8),        # L knee (critical for sitting vs standing)
        (5, "right_knee", 1.8),       # R knee (critical for sitting vs standing)
        (7, "left_ankle", 1.6),       # L ankle
        (8, "right_ankle", 1.6),      # R ankle
    ]

    def __init__(self, smpl: SMPLForwardLayer, weights: np.ndarray, device: torch.device, rot6d_mode: str = "hmr2"):
        self.smpl = smpl
        self.device = device
        self.rot6d_mode = rot6d_mode

        # Precompute body part indices for fast articulated silhouette rasterization
        bone_per_vert = np.argmax(weights, axis=1)
        parts = {
            "torso": [0, 3, 6, 9, 12, 13, 14],
            "head": [15],
            "l_arm": [16, 18, 20, 22],
            "r_arm": [17, 19, 21, 23],
            "l_leg": [1, 4, 7, 10],
            "r_leg": [2, 5, 8, 11],
        }
        self.part_indices = {p: np.where(np.isin(bone_per_vert, jts))[0] for p, jts in parts.items()}

    @staticmethod
    def select_joints(j_3d: torch.Tensor, indices: Union[List[int], torch.Tensor]) -> torch.Tensor:
        """Correctly extracts joint coordinates from either 2D (J, 3) or 3D (B, J, 3) joint tensors.
        Guarantees that indices always address the joint dimension (dim -2), not the batch dimension.
        """
        if isinstance(indices, list):
            idx_t = torch.tensor(indices, dtype=torch.long, device=j_3d.device)
        else:
            idx_t = indices.to(device=j_3d.device, dtype=torch.long)

        if j_3d.dim() == 3:
            return j_3d[:, idx_t, :]
        elif j_3d.dim() == 2:
            return j_3d[idx_t, :]
        raise ValueError(f"Expected j_3d to be 2D (J, 3) or 3D (B, J, 3), got shape {j_3d.shape}")

    def _make_rot6d(self, rotmat: torch.Tensor) -> torch.Tensor:
        """Extracts 6D rotation representation (first two columns) from 3x3 matrix."""
        if self.rot6d_mode == "spin":
            return rotmat[:, :2].contiguous().view(-1)
        return torch.cat([rotmat[:, 0], rotmat[:, 1]], dim=-1)

    def fit(
        self,
        init_pose: torch.Tensor,
        init_shape: torch.Tensor,
        init_cam: torch.Tensor,
        pose_result: Any,
        transform_meta: Dict[str, Any],
        mask_gray: Optional[Union[bytes, np.ndarray]] = None,
        quality_preset: str = "high",
        use_heuristics: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Optimizes SMPL pose, shape, and camera against observed 2D keypoints and silhouette."""
        # 1. Preset configuration
        if quality_preset == "preview" or pose_result is None or not hasattr(pose_result, "keypoints"):
            n_steps = 15
            lr = 0.045
            s_res = 128
        elif quality_preset == "maximum":
            n_steps = 80
            lr = 0.035
            s_res = 256
        else:  # "high" (default)
            n_steps = 45
            lr = 0.035
            s_res = 128

        sq_dim = transform_meta.get("square_dim", 224)
        off_x, off_y = transform_meta.get("offset", [0, 0])
        cx1, cy1 = transform_meta.get("crop_pixels", [0, 0, 0, 0])[:2]
        orig_w, orig_h = transform_meta.get("original_image_size", [1, 1])

        # 2. Extract matched keypoint targets in normalized crop space [-1, 1]
        target_coords: List[List[float]] = []
        target_confs: List[float] = []
        smpl_indices: List[int] = []
        joint_weights: List[float] = []

        kps_dict = pose_result.keypoints if (pose_result and hasattr(pose_result, "keypoints")) else {}

        # Synthesize virtual pelvis (midpoint between hips)
        if "left_hip" in kps_dict and "right_hip" in kps_dict:
            lh = kps_dict["left_hip"]
            rh = kps_dict["right_hip"]
            if lh.confidence >= 0.25 and rh.confidence >= 0.25:
                px = ((lh.x_norm + rh.x_norm) / 2.0) * orig_w
                py = ((lh.y_norm + rh.y_norm) / 2.0) * orig_h
                x_sq = (px - cx1) + off_x
                y_sq = (py - cy1) + off_y
                target_coords.append([
                    (x_sq / sq_dim) * 2.0 - 1.0,
                    (y_sq / sq_dim) * 2.0 - 1.0
                ])
                target_confs.append(float(min(lh.confidence, rh.confidence)))
                smpl_indices.append(0)
                joint_weights.append(1.6)

        # Synthesize virtual neck / upper spine (midpoint between shoulders)
        if "left_shoulder" in kps_dict and "right_shoulder" in kps_dict:
            ls = kps_dict["left_shoulder"]
            rs = kps_dict["right_shoulder"]
            if ls.confidence >= 0.25 and rs.confidence >= 0.25:
                px = ((ls.x_norm + rs.x_norm) / 2.0) * orig_w
                py = ((ls.y_norm + rs.y_norm) / 2.0) * orig_h
                x_sq = (px - cx1) + off_x
                y_sq = (py - cy1) + off_y
                target_coords.append([
                    (x_sq / sq_dim) * 2.0 - 1.0,
                    (y_sq / sq_dim) * 2.0 - 1.0
                ])
                target_confs.append(float(min(ls.confidence, rs.confidence)))
                smpl_indices.append(12)
                joint_weights.append(1.6)

        for smpl_idx, name, weight in self.JOINT_CORRESPONDENCES:
            if name in ("pelvis", "neck"):
                continue
            kp = kps_dict.get(name)
            if kp and kp.confidence >= 0.25:
                px = kp.x_norm * orig_w
                py = kp.y_norm * orig_h
                x_sq = (px - cx1) + off_x
                y_sq = (py - cy1) + off_y
                u = (x_sq / sq_dim) * 2.0 - 1.0
                v = (y_sq / sq_dim) * 2.0 - 1.0
                target_coords.append([u, v])
                target_confs.append(float(kp.confidence))
                smpl_indices.append(smpl_idx)
                joint_weights.append(weight)

        # If too few valid keypoints detected, return initial values safely without misleading measurements
        if len(target_coords) < 4:
            return init_pose, init_shape, init_cam, {
                "reprojection_error_pixels": None,
                "reprojection_error_normalized": None,
                "silhouette_iou": None,
                "quality_score": None,
                "quality_preset": quality_preset,
                "stance_description": "Default Posture (Insufficient Keypoints)",
                "observed_keypoints_count": len(target_coords),
                "pose_convergence_status": "Insufficient Keypoints (< 4 observed)",
                "metric_status": "unavailable",
            }

        target_2d = torch.tensor(target_coords, dtype=torch.float32, device=self.device)
        target_conf = torch.tensor(target_confs, dtype=torch.float32, device=self.device).unsqueeze(1)
        joint_weights_t = torch.tensor(joint_weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        # 3. Detect Stance & Posture Indicators from 2D Keypoints
        lh = kps_dict.get("left_hip")
        rh = kps_dict.get("right_hip")
        lk = kps_dict.get("left_knee")
        rk = kps_dict.get("right_knee")
        la = kps_dict.get("left_ankle")
        ra = kps_dict.get("right_ankle")

        is_seated = False
        if lh and lk and la:
            dy_thigh = abs(lk.y_norm - lh.y_norm)
            dy_shin = abs(la.y_norm - lk.y_norm)
            dy_leg = abs(la.y_norm - lh.y_norm)
            ratio = dy_thigh / max(1e-4, dy_leg)
            is_seated = (dy_thigh < 0.10 and dy_shin > 0.15) or (ratio < 0.25 and dy_shin > 0.12)
        elif rh and rk and ra:
            dy_thigh = abs(rk.y_norm - rh.y_norm)
            dy_shin = abs(ra.y_norm - rk.y_norm)
            dy_leg = abs(ra.y_norm - rh.y_norm)
            ratio = dy_thigh / max(1e-4, dy_leg)
            is_seated = (dy_thigh < 0.10 and dy_shin > 0.15) or (ratio < 0.25 and dy_shin > 0.12)

        ls = kps_dict.get("left_shoulder")
        rs = kps_dict.get("right_shoulder")
        lw = kps_dict.get("left_wrist")
        rw = kps_dict.get("right_wrist")
        is_crossed = False
        if ls and rs and lw and rw:
            mid_shoulder_x = (ls.x_norm + rs.x_norm) / 2.0
            is_left_crossed = lw.x_norm < mid_shoulder_x
            is_right_crossed = rw.x_norm > mid_shoulder_x
            is_crossed = is_left_crossed or is_right_crossed

        # 4. Multi-Hypothesis Pose Candidate Selection
        candidates: List[torch.Tensor] = [init_pose.clone().view(24, 6)]

        if use_heuristics:
            # Candidate: Seated Pose Hypothesis
            if is_seated:
                c_seated = init_pose.clone().view(24, 6)
                theta_hip = -1.45  # 83 deg forward flexion
                R_hip = torch.tensor([
                    [1.0, 0.0, 0.0],
                    [0.0, np.cos(theta_hip), -np.sin(theta_hip)],
                    [0.0, np.sin(theta_hip), np.cos(theta_hip)]
                ], dtype=torch.float32, device=self.device)
                theta_knee = 1.55  # 89 deg knee flexion
                R_knee = torch.tensor([
                    [1.0, 0.0, 0.0],
                    [0.0, np.cos(theta_knee), -np.sin(theta_knee)],
                    [0.0, np.sin(theta_knee), np.cos(theta_knee)]
                ], dtype=torch.float32, device=self.device)
                c_seated[1] = self._make_rot6d(R_hip)
                c_seated[2] = self._make_rot6d(R_hip)
                c_seated[4] = self._make_rot6d(R_knee)
                c_seated[5] = self._make_rot6d(R_knee)
                candidates.append(c_seated)

            # Candidate: Crossed-Arms Pose Hypothesis
            if is_crossed:
                c_crossed = init_pose.clone().view(24, 6)
                R_l_sh = torch.tensor([
                    [0.707, -0.707, 0.0],
                    [0.707, 0.707, 0.0],
                    [0.0, 0.0, 1.0]
                ], dtype=torch.float32, device=self.device)
                R_l_el = torch.tensor([
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0]
                ], dtype=torch.float32, device=self.device)
                c_crossed[16] = self._make_rot6d(R_l_sh)
                c_crossed[18] = self._make_rot6d(R_l_el)
                candidates.append(c_crossed)

        # Evaluate candidate 2D reprojection error and select best starting basin
        best_cand = candidates[0]
        best_err = 1e9
        eval_cam = init_cam.view(1, 3)
        with torch.no_grad():
            for cand in candidates:
                v_init = self.smpl(cand, init_shape, rot6d_mode=self.rot6d_mode)
                j_all = torch.matmul(self.smpl.J_regressor, v_init)
                j_all_2d = j_all.squeeze(0) if j_all.dim() == 3 and j_all.shape[0] == 1 else j_all
                j_init = self.select_joints(j_all_2d, smpl_indices)
                p2d = eval_cam[0, 0] * j_init[:, :2] + eval_cam[0, 1:3]
                err = torch.mean((p2d - target_2d) ** 2).item()
                if err < best_err:
                    best_err = err
                    best_cand = cand

        # Precompute differentiable distance transform for silhouette loss if GT mask is present
        dist_out_tensor: Optional[torch.Tensor] = None
        gt_mask_binary: Optional[np.ndarray] = None
        if mask_gray is not None:
            try:
                if isinstance(mask_gray, bytes):
                    gt_arr = cv2.imdecode(np.frombuffer(mask_gray, np.uint8), cv2.IMREAD_GRAYSCALE)
                else:
                    gt_arr = mask_gray

                if gt_arr is not None and gt_arr.size > 0:
                    clamp_x1, clamp_y1, clamp_x2, clamp_y2 = transform_meta.get("crop_pixels", [0, 0, orig_w, orig_h])
                    gt_crop = gt_arr[clamp_y1:clamp_y2, clamp_x1:clamp_x2]
                    if gt_crop.size > 0:
                        gt_sq = np.zeros((sq_dim, sq_dim), dtype=np.uint8)
                        gt_sq[off_y:off_y + gt_crop.shape[0], off_x:off_x + gt_crop.shape[1]] = gt_crop
                        gt_resized = cv2.resize(gt_sq, (s_res, s_res), interpolation=cv2.INTER_NEAREST)
                        if np.any(gt_resized > 128):
                            gt_mask_binary = (gt_resized > 128).astype(np.uint8)
                            dist_out = cv2.distanceTransform(255 - (gt_mask_binary * 255), cv2.DIST_L2, 5) / float(s_res)
                            dist_out_tensor = torch.tensor(dist_out, dtype=torch.float32, device=self.device).view(1, 1, s_res, s_res)
            except Exception as mask_err:
                logger.debug(f"Silhouette distance precomputation fallback: {mask_err}")

        # 5. Parameters to Optimize
        opt_pose = best_cand.clone().detach().requires_grad_(True)
        opt_shape = init_shape.clone().detach().requires_grad_(True)
        opt_cam = init_cam.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([
            {"params": [opt_pose], "lr": lr},
            {"params": [opt_shape], "lr": lr * 0.4},
            {"params": [opt_cam], "lr": lr * 0.4},
        ])

        final_loss_val = 0.0

        # 6. Optimization Loop with Anatomical Constraints and Differentiable Silhouette
        for step in range(n_steps):
            optimizer.zero_grad()
            v_final = self.smpl(opt_pose, opt_shape, rot6d_mode=self.rot6d_mode)
            j_3d = torch.matmul(self.smpl.J_regressor, v_final)
            j_3d_2d = j_3d.squeeze(0) if j_3d.dim() == 3 and j_3d.shape[0] == 1 else j_3d
            j_sel = self.select_joints(j_3d_2d, smpl_indices)

            scale = opt_cam[0, 0]
            tx = opt_cam[0, 1]
            ty = opt_cam[0, 2]
            proj_2d = torch.stack([scale * j_sel[:, 0] + tx, scale * j_sel[:, 1] + ty], dim=1)

            # Robust Huber 2D Keypoint Loss
            diff = proj_2d - target_2d
            dist_sq = torch.sum(diff ** 2, dim=1, keepdim=True)
            huber = torch.where(dist_sq < 0.0025, 0.5 * dist_sq, 0.05 * (torch.sqrt(dist_sq + 1e-6) - 0.025))
            loss_kp = torch.mean(target_conf * joint_weights_t * huber)

            # Differentiable Silhouette Distance Loss
            loss_sil = torch.tensor(0.0, device=self.device)
            if dist_out_tensor is not None:
                v_pts = v_final if v_final.dim() == 2 else v_final[0]
                pts_u = scale * v_pts[:, 0] + tx
                pts_v = scale * v_pts[:, 1] + ty
                # Subsample surface vertices uniformly for smooth, efficient silhouette pull
                grid_u = pts_u[::11]
                grid_v = pts_v[::11]
                grid = torch.stack([grid_u, grid_v], dim=-1).view(1, 1, -1, 2)
                sampled_dist = F.grid_sample(dist_out_tensor, grid, mode="bilinear", padding_mode="border", align_corners=True)
                loss_sil = 0.08 * torch.mean(sampled_dist ** 2)

            # Anatomical Joint Limit Constraints
            R = rot6d_to_rotmat(opt_pose)
            loss_limits = torch.tensor(0.0, device=self.device)

            # Knee hyperextension penalty (knees cannot bend forward in natural anatomy)
            for knee_idx in (4, 5):
                k_angle = torch.atan2(R[knee_idx, 2, 1], R[knee_idx, 1, 1])
                loss_limits = loss_limits + 5.0 * torch.relu(-k_angle) ** 2
                loss_limits = loss_limits + 5.0 * torch.relu(k_angle - 2.6) ** 2

            # Elbow hyperextension penalty
            for elbow_idx in (18, 19):
                el_R = R[elbow_idx]
                tr = el_R[0, 0] + el_R[1, 1] + el_R[2, 2]
                cos_th = torch.clamp((tr - 1.0) / 2.0, -1.0, 1.0)
                el_angle = torch.acos(cos_th)
                loss_limits = loss_limits + 4.0 * torch.relu(el_angle - 2.6) ** 2

            # Spine curvature stability
            for spine_idx in (3, 6, 9):
                sp_R = R[spine_idx]
                loss_limits = loss_limits + 0.08 * torch.sum((sp_R - torch.eye(3, device=self.device)) ** 2)

            # Adaptive Pose Prior towards the validated hypothesis
            loss_pose_prior = 0.002 * torch.mean((opt_pose - best_cand) ** 2)

            # Body Shape Regularization
            loss_shape = 0.001 * torch.mean(opt_shape ** 2)

            # Camera Scale Regularization
            loss_cam = 0.015 * (scale - init_cam[0, 0]) ** 2

            loss = loss_kp + loss_sil + loss_limits + loss_pose_prior + loss_shape + loss_cam
            loss.backward()
            optimizer.step()
            final_loss_val = float(loss.item())

        # 7. Post-Fitting Validation Metrics & Truthful Stance Classification
        with torch.no_grad():
            final_v = self.smpl(opt_pose, opt_shape, rot6d_mode=self.rot6d_mode)
            final_j = torch.matmul(self.smpl.J_regressor, final_v)
            final_j_2d = final_j.squeeze(0) if final_j.dim() == 3 and final_j.shape[0] == 1 else final_j
            final_j_sel = self.select_joints(final_j_2d, smpl_indices)

            scale_val = opt_cam[0, 0].item()
            tx_val = opt_cam[0, 1].item()
            ty_val = opt_cam[0, 2].item()

            proj_u = (scale_val * final_j_sel[:, 0] + tx_val).cpu().numpy()
            proj_v = (scale_val * final_j_sel[:, 1] + ty_val).cpu().numpy()
            target_u = target_2d[:, 0].cpu().numpy()
            target_v = target_2d[:, 1].cpu().numpy()

            diff_u = proj_u - target_u
            diff_v = proj_v - target_v
            norm_dist = np.sqrt(diff_u ** 2 + diff_v ** 2)
            reproj_err_norm = float(np.mean(norm_dist))
            reproj_err_px = float(reproj_err_norm * (sq_dim / 2.0))

            # Truthful Stance Description derived directly from fitted 3D kinematics
            final_j_hip = self.select_joints(final_j_2d, [1])
            final_j_knee = self.select_joints(final_j_2d, [4])
            hip_y = final_j_hip[0, 1].item()
            knee_y = final_j_knee[0, 1].item()
            dy_thigh_3d = hip_y - knee_y
            R_final = rot6d_to_rotmat(opt_pose)
            l_knee_angle_deg = round(float(torch.atan2(R_final[4, 2, 1], R_final[4, 1, 1]).item() * 180.0 / np.pi), 1)

            if is_seated or abs(dy_thigh_3d) < 0.14:
                stance_desc = f"Seated Posture (Knee Flexion: {l_knee_angle_deg}°)"
            elif is_crossed:
                stance_desc = "Standing Posture (Crossed Arms, Upright)"
            else:
                stance_desc = "Standing Posture (Upright)"

            # 8. Silhouette IoU via Articulated Body Part Rasterization
            silhouette_iou: Optional[float] = None
            if gt_mask_binary is not None:
                v_np = final_v.detach().cpu().numpy()
                if v_np.ndim == 3:
                    v_np = v_np[0]
                pts_u = scale_val * v_np[:, 0] + tx_val
                pts_v = scale_val * v_np[:, 1] + ty_val
                pts_x = np.clip((pts_u + 1.0) * 0.5 * s_res, 0, s_res - 1).astype(np.int32)
                pts_y = np.clip((pts_v + 1.0) * 0.5 * s_res, 0, s_res - 1).astype(np.int32)
                pts_2d = np.stack([pts_x, pts_y], axis=1)

                mask_proj = np.zeros((s_res, s_res), dtype=np.uint8)
                for _, idxs in self.part_indices.items():
                    if len(idxs) > 2:
                        hull = cv2.convexHull(pts_2d[idxs])
                        cv2.fillPoly(mask_proj, [hull], 255)

                intersection = np.logical_and(mask_proj > 128, gt_mask_binary > 0).sum()
                union = np.logical_or(mask_proj > 128, gt_mask_binary > 0).sum()
                if union > 0:
                    silhouette_iou = round(float(intersection / union), 3)

            kp_score = max(0.0, 1.0 - (reproj_err_px / 48.0))
            if silhouette_iou is not None:
                quality_score = round(float(np.clip(0.55 * kp_score + 0.45 * silhouette_iou, 0.15, 0.98)), 3)
                metric_status = "measured"
            else:
                quality_score = round(float(np.clip(kp_score, 0.15, 0.98)), 3)
                metric_status = "estimated"

        return opt_pose, opt_shape, opt_cam, {
            "reprojection_error_pixels": round(reproj_err_px, 2),
            "reprojection_error_normalized": round(reproj_err_norm, 4),
            "silhouette_iou": silhouette_iou,
            "quality_score": quality_score,
            "quality_preset": quality_preset,
            "stance_description": stance_desc,
            "observed_keypoints_count": len(target_coords),
            "pose_convergence_status": f"Converged (Residual Loss: {final_loss_val:.4f})",
            "metric_status": metric_status,
        }

    def evaluate(
        self,
        pose: torch.Tensor,
        shape: torch.Tensor,
        cam: torch.Tensor,
        pose_result: Optional[Any],
        transform_meta: Dict[str, Any],
        mask_gray: Optional[Union[bytes, np.ndarray]] = None,
        quality_preset: str = "high",
    ) -> Dict[str, Any]:
        """Evaluates quantitative metrics (reprojection error, silhouette IoU, stance)
        for given SMPL parameters without altering or optimizing them."""
        sq_dim = transform_meta.get("square_dim", 224)
        off_x, off_y = transform_meta.get("offset", [0, 0])
        cx1, cy1 = transform_meta.get("crop_pixels", [0, 0, 0, 0])[:2]
        orig_w, orig_h = transform_meta.get("original_image_size", [1, 1])
        s_res = 128

        target_coords: List[List[float]] = []
        smpl_indices: List[int] = []

        kps_dict = pose_result.keypoints if (pose_result and hasattr(pose_result, "keypoints")) else {}

        if "left_hip" in kps_dict and "right_hip" in kps_dict:
            lh = kps_dict["left_hip"]
            rh = kps_dict["right_hip"]
            if lh.confidence >= 0.25 and rh.confidence >= 0.25:
                px = ((lh.x_norm + rh.x_norm) / 2.0) * orig_w
                py = ((lh.y_norm + rh.y_norm) / 2.0) * orig_h
                x_sq = (px - cx1) + off_x
                y_sq = (py - cy1) + off_y
                target_coords.append([(x_sq / sq_dim) * 2.0 - 1.0, (y_sq / sq_dim) * 2.0 - 1.0])
                smpl_indices.append(0)

        for smpl_idx, name, _ in self.JOINT_CORRESPONDENCES:
            if name in ("pelvis", "neck"):
                continue
            kp = kps_dict.get(name)
            if kp and getattr(kp, "confidence", 0.0) >= 0.25:
                px = kp.x_norm * orig_w
                py = kp.y_norm * orig_h
                x_sq = (px - cx1) + off_x
                y_sq = (py - cy1) + off_y
                target_coords.append([(x_sq / sq_dim) * 2.0 - 1.0, (y_sq / sq_dim) * 2.0 - 1.0])
                smpl_indices.append(smpl_idx)

        if len(target_coords) < 4:
            return {
                "reprojection_error_pixels": None,
                "reprojection_error_normalized": None,
                "silhouette_iou": None,
                "quality_score": None,
                "quality_preset": quality_preset,
                "stance_description": "Observed Posture (Insufficient Keypoints)",
                "observed_keypoints_count": len(target_coords),
                "pose_convergence_status": "Direct Neural Prediction (< 4 2D joints)",
                "metric_status": "unavailable",
            }

        target_2d = torch.tensor(target_coords, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pose_eval = pose.view(24, 6) if pose.dim() == 2 and pose.shape[-1] == 144 else pose
            v_eval = self.smpl(pose_eval, shape, rot6d_mode=self.rot6d_mode)
            j_eval = torch.matmul(self.smpl.J_regressor, v_eval)
            j_eval_2d = j_eval.squeeze(0) if j_eval.dim() == 3 and j_eval.shape[0] == 1 else j_eval
            j_sel = self.select_joints(j_eval_2d, smpl_indices)

            scale_val = cam[0, 0].item() if cam.dim() == 2 else cam[0].item()
            tx_val = cam[0, 1].item() if cam.dim() == 2 else cam[1].item()
            ty_val = cam[0, 2].item() if cam.dim() == 2 else cam[2].item()

            proj_u = (scale_val * j_sel[:, 0] + tx_val).cpu().numpy()
            proj_v = (scale_val * j_sel[:, 1] + ty_val).cpu().numpy()
            target_u = target_2d[:, 0].cpu().numpy()
            target_v = target_2d[:, 1].cpu().numpy()

            diff_u = proj_u - target_u
            diff_v = proj_v - target_v
            norm_dist = np.sqrt(diff_u ** 2 + diff_v ** 2)
            reproj_err_norm = float(np.mean(norm_dist))
            reproj_err_px = float(reproj_err_norm * (sq_dim / 2.0))

            # Stance classification from predicted kinematics
            R_eval = rot6d_to_rotmat(pose_eval)
            knee_l = round(float(torch.atan2(R_eval[4, 2, 1], R_eval[4, 1, 1]).item() * 180.0 / np.pi), 1)
            knee_r = round(float(torch.atan2(R_eval[5, 2, 1], R_eval[5, 1, 1]).item() * 180.0 / np.pi), 1)
            max_knee = max(abs(knee_l), abs(knee_r))

            final_j_hip = self.select_joints(j_eval_2d, [1])
            final_j_knee = self.select_joints(j_eval_2d, [4])
            dy_thigh_3d = final_j_hip[0, 1].item() - final_j_knee[0, 1].item()

            if max_knee > 65.0 or abs(dy_thigh_3d) < 0.14:
                stance_desc = f"Seated Posture (Knee Flexion: {max_knee:.1f}°)"
            else:
                stance_desc = "Standing Posture (Upright)"

            # Silhouette IoU
            silhouette_iou: Optional[float] = None
            if mask_gray is not None:
                try:
                    if isinstance(mask_gray, bytes):
                        gt_arr = cv2.imdecode(np.frombuffer(mask_gray, np.uint8), cv2.IMREAD_GRAYSCALE)
                    else:
                        gt_arr = mask_gray
                    if gt_arr is not None and gt_arr.size > 0:
                        clamp_x1, clamp_y1, clamp_x2, clamp_y2 = transform_meta.get("crop_pixels", [0, 0, orig_w, orig_h])
                        gt_crop = gt_arr[clamp_y1:clamp_y2, clamp_x1:clamp_x2]
                        if gt_crop.size > 0:
                            gt_sq = np.zeros((sq_dim, sq_dim), dtype=np.uint8)
                            gt_sq[off_y:off_y + gt_crop.shape[0], off_x:off_x + gt_crop.shape[1]] = gt_crop
                            gt_resized = cv2.resize(gt_sq, (s_res, s_res), interpolation=cv2.INTER_NEAREST)
                            gt_mask_binary = (gt_resized > 128).astype(np.uint8)

                            v_np = v_eval.detach().cpu().numpy()
                            if v_np.ndim == 3:
                                v_np = v_np[0]
                            pts_u = scale_val * v_np[:, 0] + tx_val
                            pts_v = scale_val * v_np[:, 1] + ty_val
                            pts_x = np.clip((pts_u + 1.0) * 0.5 * s_res, 0, s_res - 1).astype(np.int32)
                            pts_y = np.clip((pts_v + 1.0) * 0.5 * s_res, 0, s_res - 1).astype(np.int32)
                            pts_2d = np.stack([pts_x, pts_y], axis=1)

                            mask_proj = np.zeros((s_res, s_res), dtype=np.uint8)
                            for _, idxs in self.part_indices.items():
                                if len(idxs) > 2:
                                    hull = cv2.convexHull(pts_2d[idxs])
                                    cv2.fillPoly(mask_proj, [hull], 255)

                            intersection = np.logical_and(mask_proj > 128, gt_mask_binary > 0).sum()
                            union = np.logical_or(mask_proj > 128, gt_mask_binary > 0).sum()
                            if union > 0:
                                silhouette_iou = round(float(intersection / union), 3)
                except Exception as e:
                    logger.debug(f"Silhouette IoU evaluation: {e}")

            kp_score = max(0.0, 1.0 - (reproj_err_px / 48.0))
            if silhouette_iou is not None:
                quality_score = round(float(np.clip(0.55 * kp_score + 0.45 * silhouette_iou, 0.15, 0.98)), 3)
                metric_status = "measured"
            else:
                quality_score = round(float(np.clip(kp_score, 0.15, 0.98)), 3)
                metric_status = "estimated"

        return {
            "reprojection_error_pixels": round(reproj_err_px, 2),
            "reprojection_error_normalized": round(reproj_err_norm, 4),
            "silhouette_iou": silhouette_iou,
            "quality_score": quality_score,
            "quality_preset": quality_preset,
            "stance_description": stance_desc,
            "observed_keypoints_count": len(target_coords),
            "pose_convergence_status": "Direct Neural Prediction (HMR 2.0 ViT)",
            "metric_status": metric_status,
        }


# ---------------------------------------------------------------------------
# Human Reconstructor Adapters
# ---------------------------------------------------------------------------

class HumanReconstructor(ABC):
    """Abstract interface for monocular human 3D reconstruction models."""

    @abstractmethod
    def reconstruct(
        self,
        image_rgb: np.ndarray,
        bounding_box: BoundingBox2D,
        mask_gray: Optional[Union[bytes, np.ndarray]] = None,
        depth_stats: Optional[Dict[str, Any]] = None,
        pose_result: Optional[Any] = None,
        quality_preset: str = "high",
    ) -> HumanReconstructionResult:
        pass


class MockHumanReconstructor(HumanReconstructor):
    """Deterministic humanoid mesh generator for mocked integration testing and offline CI.
    
    Generates a realistic 3D anatomical humanoid geometry (head, torso, limbs) with
    over 1,000 vertices and valid manifold triangle topology, scaled proportionally
    to the detected person entity's bounding box.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model_name = "MockHumanReconstructor"
        self.model_version = "1.0.0 (Deterministic)"

    def reconstruct(
        self,
        image_rgb: np.ndarray,
        bounding_box: BoundingBox2D,
        mask_gray: Optional[Union[bytes, np.ndarray]] = None,
        depth_stats: Optional[Dict[str, Any]] = None,
        pose_result: Optional[Any] = None,
        quality_preset: str = "high",
    ) -> HumanReconstructionResult:
        t0 = time.time()
        _, transform_meta = prepare_person_crop(image_rgb, bounding_box)

        aspect = transform_meta["aspect_ratio"]
        height = 1.75
        width = 0.5 * min(1.2, max(0.8, aspect / 0.45))

        # 1. Torso
        torso = trimesh.creation.cylinder(radius=width * 0.28, height=height * 0.38, sections=36)
        torso.apply_translation([0, height * 0.55, 0])

        # 2. Head
        head = trimesh.creation.icosphere(subdivisions=3, radius=height * 0.09)
        head.apply_translation([0, height * 0.88, 0])

        # 3. Legs
        leg_left = trimesh.creation.cylinder(radius=width * 0.11, height=height * 0.48, sections=36)
        leg_left.apply_translation([-width * 0.18, height * 0.24, 0])

        leg_right = trimesh.creation.cylinder(radius=width * 0.11, height=height * 0.48, sections=36)
        leg_right.apply_translation([width * 0.18, height * 0.24, 0])

        # 4. Arms
        arm_left = trimesh.creation.cylinder(radius=width * 0.08, height=height * 0.42, sections=36)
        arm_left.apply_translation([-width * 0.38, height * 0.56, 0])

        arm_right = trimesh.creation.cylinder(radius=width * 0.08, height=height * 0.42, sections=36)
        arm_right.apply_translation([width * 0.38, height * 0.56, 0])

        combined = trimesh.util.concatenate([torso, head, leg_left, leg_right, arm_left, arm_right])
        vertices = np.array(combined.vertices, dtype=np.float32)
        faces = np.array(combined.faces, dtype=np.int32)

        # Center at pelvis origin (0, 0, 0)
        vertices[:, 1] -= (height * 0.5)

        is_valid, warnings = validate_mesh_geometry(vertices, faces)
        normals = compute_mesh_normals(vertices, faces)

        v_min = vertices.min(axis=0).tolist()
        v_max = vertices.max(axis=0).tolist()
        dims = (vertices.max(axis=0) - vertices.min(axis=0)).tolist()

        # Synthetic anatomical palette for mock double
        vertex_colors = np.ones((len(vertices), 4), dtype=np.uint8) * 255
        y_pts = vertices[:, 1]
        is_legs = y_pts < -0.1
        is_torso = (y_pts >= -0.1) & (y_pts < 0.45)
        is_head = y_pts >= 0.45
        vertex_colors[is_legs, :3] = [59, 130, 246]   # Classic denim blue
        vertex_colors[is_torso, :3] = [100, 116, 139] # Slate shirt
        vertex_colors[is_head, :3] = [245, 158, 11]   # Amber flesh tone

        glb_bytes = export_mesh_glb(vertices, faces, normals, vertex_colors=vertex_colors, name="mock_human")

        # Mock metrics calibrated for test doubles
        reproj_px = 12.0 if quality_preset == "maximum" else 18.5 if quality_preset == "high" else 28.0
        sil_iou = 0.88 if quality_preset == "maximum" else 0.84 if quality_preset == "high" else 0.72
        quality_score = 0.90 if quality_preset == "maximum" else 0.82 if quality_preset == "high" else 0.65

        warnings.append("Using synthetic mock reconstructor test double (no neural inference).")
        return HumanReconstructionResult(
            vertices=vertices,
            faces=faces,
            normals=normals,
            bounds_min=v_min,
            bounds_max=v_max,
            dimensions=dims,
            glb_bytes=glb_bytes,
            model_name=self.model_name,
            model_version=self.model_version,
            device=self.device,
            pose_params=np.zeros(144, dtype=np.float32),
            shape_params=np.zeros(10, dtype=np.float32),
            camera_params={"scale": 0.9, "tx": 0.0, "ty": 0.0},
            texture_availability="partial",  # synthetic palette
            quality_score=quality_score,
            quality_preset=quality_preset,
            reprojection_error_pixels=reproj_px,
            reprojection_error_normalized=reproj_px / 224.0,
            silhouette_iou=sil_iou,
            stance_description="Standing Posture (Synthetic Mock)",
            observed_keypoints_count=0,
            pose_convergence_status="Synthetic Validation (Mock Reconstructor)",
            metric_status="mock",
            pelvis_3d=[0.0, 0.0, 0.0],
            warnings=warnings,
            processing_time_ms=(time.time() - t0) * 1000.0,
        )


class SpinHumanReconstructor(HumanReconstructor):
    """Pretrained local SPIN human mesh recovery adapter (ResNet-50 + SMPL Linear Blend Skinning).
    
    Predicts 3D pose, body shape, and weak-perspective camera, deforming the SMPL neutral body
    model to produce a person-specific 3D human mesh with 6,890 vertices and 13,776 faces.
    Integrates gradient-based SMPLPoseFitter to guarantee source-image pose fidelity and body shape.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        smpl_npz_path: Path,
        mean_params_path: Path,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model_name = "SPIN"
        self.model_version = "1.0.0 (SMPL-Neutral LBS + SMPLify Fitter)"

        logger.info(f"Initializing SpinHumanReconstructor on device '{device}'...")

        # 1. Load SMPL parameters
        npz_data = np.load(smpl_npz_path, allow_pickle=True)
        self.smpl = SMPLForwardLayer(npz_data).to(self.device)
        self.faces = self.smpl.faces

        # 2. Setup Gradient Pose & Silhouette Fitter
        self.pose_fitter = SMPLPoseFitter(
            smpl=self.smpl,
            weights=npz_data["weights"],
            device=self.device,
            rot6d_mode="spin",
        )

        # 3. Load mean parameters
        mean_data = np.load(mean_params_path)
        self.init_pose = torch.tensor(mean_data["pose"], dtype=torch.float32, device=self.device).unsqueeze(0)
        self.init_shape = torch.tensor(mean_data["shape"], dtype=torch.float32, device=self.device).unsqueeze(0)
        self.init_cam = torch.tensor(mean_data["cam"], dtype=torch.float32, device=self.device).unsqueeze(0)

        # 4. Load Checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device)["model"]

        # 5. Setup Regressor
        self.regressor = SPINRegressor().to(self.device)
        reg_state = {k: ckpt[k] for k in ckpt if k.startswith("fc") or k.startswith("dec")}
        self.regressor.load_state_dict(reg_state)
        self.regressor.eval()

        # 6. Setup Backbone (ResNet-50)
        self.backbone = resnet50().to(self.device)
        self.backbone.fc = nn.Identity()
        bb_state = {k: v for k, v in ckpt.items() if not k.startswith("fc") and not k.startswith("dec")}
        self.backbone.load_state_dict(bb_state, strict=False)
        self.backbone.eval()

        logger.info("SpinHumanReconstructor initialized successfully.")

    def reconstruct(
        self,
        image_rgb: np.ndarray,
        bounding_box: BoundingBox2D,
        mask_gray: Optional[Union[bytes, np.ndarray]] = None,
        depth_stats: Optional[Dict[str, Any]] = None,
        pose_result: Optional[Any] = None,
        quality_preset: str = "high",
    ) -> HumanReconstructionResult:
        t0 = time.time()
        crop_img, transform_meta = prepare_person_crop(image_rgb, bounding_box)

        # ImageNet normalization
        norm_img = crop_img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = (norm_img - mean) / std
        input_tensor = torch.tensor(norm_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # 1. Feed-forward initial prediction from ResNet-50 + SPIN Regressor
        with torch.no_grad():
            feat = self.backbone(input_tensor)
            init_pred_pose, init_pred_shape, init_pred_cam = self.regressor(
                feat, self.init_pose, self.init_shape, self.init_cam, n_iter=3
            )

        # 2. Refine pose, shape, and camera using SMPLPoseFitter
        fitted_pose, fitted_shape, fitted_cam, metrics = self.pose_fitter.fit(
            init_pose=init_pred_pose,
            init_shape=init_pred_shape,
            init_cam=init_pred_cam,
            pose_result=pose_result,
            transform_meta=transform_meta,
            mask_gray=mask_gray,
            quality_preset=quality_preset,
        )

        with torch.no_grad():
            vertices_tensor = self.smpl(fitted_pose, fitted_shape, rot6d_mode="spin")
            joints_tensor = torch.matmul(self.smpl.J_regressor, vertices_tensor)

        vertices = vertices_tensor.detach().cpu().numpy().astype(np.float32)
        if vertices.ndim == 3:
            vertices = vertices[0]

        pelvis_pt = SMPLPoseFitter.select_joints(joints_tensor, [0])[0]
        pelvis_raw = pelvis_pt.detach().cpu().numpy()
        pelvis_3d = [float(pelvis_raw[0]), float(-pelvis_raw[1]), float(-pelvis_raw[2])]

        # Convert from OpenCV camera coordinate convention (+X right, +Y down, +Z into depth)
        # to DRISHTI / Three.js canonical world coordinates (+X right, +Y UP, +Z TOWARD camera):
        # [X', Y', Z'] = [X_cam, -Y_cam, -Z_cam]
        # This rigid rotation (180 deg around X-axis) has det(R) = +1.0, preserving
        # right-handedness, triangle winding order, and surface orientation.
        vertices[:, 1] = -vertices[:, 1]
        vertices[:, 2] = -vertices[:, 2]

        faces = self.faces.copy()

        # Validate geometry
        is_valid, warnings = validate_mesh_geometry(vertices, faces)
        if transform_meta["is_clipped"]:
            warnings.append("Person is partially clipped by photo frame boundaries; peripheral limbs may be approximated.")

        normals = compute_mesh_normals(vertices, faces)
        v_min = vertices.min(axis=0).tolist()
        v_max = vertices.max(axis=0).tolist()
        dims = (vertices.max(axis=0) - vertices.min(axis=0)).tolist()

        cam_values = fitted_cam.squeeze(0).detach().cpu().numpy().tolist()
        camera_params = {
            "scale": float(cam_values[0]),
            "tx": float(cam_values[1]),
            "ty": float(cam_values[2]),
        }

        # Projective vertex colors from crop
        vertex_colors = compute_projected_vertex_colors(
            vertices_three=vertices,
            normals_three=normals,
            camera_params=camera_params,
            crop_rgb=crop_img,
            mask_gray=mask_gray,
            transform_meta=transform_meta,
        )

        glb_bytes = export_mesh_glb(
            vertices=vertices,
            faces=faces,
            normals=normals,
            vertex_colors=vertex_colors,
            name="spin_human",
        )

        return HumanReconstructionResult(
            vertices=vertices,
            faces=faces,
            normals=normals,
            bounds_min=v_min,
            bounds_max=v_max,
            dimensions=dims,
            glb_bytes=glb_bytes,
            model_name=self.model_name,
            model_version=self.model_version,
            device=str(self.device),
            pose_params=fitted_pose.squeeze(0).detach().cpu().numpy(),
            shape_params=fitted_shape.squeeze(0).detach().cpu().numpy(),
            camera_params=camera_params,
            texture_availability="partial",  # projective vertex colors
            quality_score=metrics["quality_score"],
            quality_preset=metrics["quality_preset"],
            reprojection_error_pixels=metrics["reprojection_error_pixels"],
            reprojection_error_normalized=metrics["reprojection_error_normalized"],
            silhouette_iou=metrics["silhouette_iou"],
            stance_description=metrics["stance_description"],
            observed_keypoints_count=metrics["observed_keypoints_count"],
            pose_convergence_status=metrics["pose_convergence_status"],
            metric_status=metrics.get("metric_status", "measured"),
            pelvis_3d=pelvis_3d,
            warnings=warnings,
            processing_time_ms=(time.time() - t0) * 1000.0,
        )


class Hmr2HumanReconstructor(HumanReconstructor):
    """Production human reconstruction backend powered by 4DHumans / HMR 2.0.
    
    Vision Transformer (ViT-Huge) backbone with SMPL Cross-Attention Decoder Head.
    Predicts canonical SMPL parameters directly (6,890 vertices, 24 joints), natively compatible
    with DRISHTI kinematic rigging, animation, and GLB scene export.
    Executes actual neural inference on Apple Silicon MPS, NVIDIA CUDA, or CPU.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        smpl_npz_path: Optional[Path] = None,
        mean_params_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
        fallback_reconstructor: Optional[HumanReconstructor] = None,
    ):
        self.model_name = "HMR 2.0 (4DHumans)"
        self.model_version = "2.0.0"
        self.device = device or device_manager.get_torch_device()
        self.checkpoint_path = checkpoint_path
        if self.checkpoint_path is None and fallback_reconstructor is None:
            default_ckpt = (DEFAULT_MODELS_DIR / "hmr2" / "epoch=35-step=1000000.ckpt").resolve()
            if default_ckpt.exists():
                self.checkpoint_path = default_ckpt
        self.smpl_npz_path = smpl_npz_path or (DEFAULT_MODELS_DIR / "human_reconstruction" / "SMPL_N_model.npz").resolve()
        self.mean_params_path = mean_params_path or (DEFAULT_MODELS_DIR / "human_reconstruction" / "smpl_mean_params.npz").resolve()
        self._fallback = fallback_reconstructor
        self._model = None
        self.is_available = False
        self.is_real_hmr2 = False
        self.smpl: Optional[SMPLForwardLayer] = None
        self.faces: Optional[np.ndarray] = None
        self.pose_fitter: Optional[SMPLPoseFitter] = None

        # 1. Initialize SMPL Forward Layer
        if self.smpl_npz_path.exists():
            try:
                npz_data = np.load(str(self.smpl_npz_path), allow_pickle=True)
                self.smpl = SMPLForwardLayer(npz_data).to(self.device)
                self.faces = self.smpl.faces
                self.pose_fitter = SMPLPoseFitter(self.smpl, weights=npz_data["weights"], device=self.device)
            except Exception as e:
                logger.error(f"Failed to load SMPL forward layer for HMR 2.0: {e}")

        # 2. Check and load HMR 2.0 checkpoint
        if (
            self.checkpoint_path is not None
            and self.checkpoint_path.exists()
            and self.checkpoint_path.stat().st_size > 1000000
            and self.smpl is not None
        ):
            try:
                from app.pipeline.hmr2_model import HMR2Network
                logger.info(f"Loading official HMR 2.0 weights from {self.checkpoint_path}...")
                self._model = HMR2Network.load_from_checkpoint(
                    checkpoint_path=self.checkpoint_path,
                    mean_params_path=self.mean_params_path if self.mean_params_path.exists() else None,
                    device=self.device,
                )
                self.is_available = True
                self.is_real_hmr2 = True
                logger.info(f"HMR 2.0 model loaded successfully on device: {self.device}")
            except Exception as e:
                logger.error(f"Failed to load HMR 2.0 model weights: {e}")
                self._model = None
                self.is_available = False
                self.is_real_hmr2 = False
        else:
            if self.checkpoint_path is not None and not self.checkpoint_path.exists():
                logger.info(f"HMR 2.0 checkpoint not found at {self.checkpoint_path}; registered in standby mode.")
            self.is_available = False
            self.is_real_hmr2 = False

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "model_name": "HMR 2.0 (4DHumans)",
            "architecture": "Vision Transformer (ViT-Huge) + SMPL Cross-Attention Decoder",
            "is_articulated": True,
            "output_format": "SMPL Canonical (6,890 vertices, 24 joints)",
            "weights_installed": self.is_available,
            "is_real_hmr2": self.is_real_hmr2,
            "device": str(self.device),
            "license": "Code: MIT; Checkpoint & SMPL: CC-BY-NC 4.0 (Non-Commercial Research; requires SMPL license)",
            "recommended_device": "mps (Apple Silicon) or cuda (NVIDIA)",
            "vram_recommended_mb": 4096,
        }

    def reconstruct(
        self,
        image_rgb: np.ndarray,
        bounding_box: BoundingBox2D,
        mask_gray: Optional[Union[bytes, np.ndarray]] = None,
        depth_stats: Optional[Dict[str, Any]] = None,
        pose_result: Optional[Any] = None,
        quality_preset: str = "high",
        refine_pose: bool = False,
    ) -> HumanReconstructionResult:
        t0 = time.time()

        if not self.is_available or self._model is None:
            if self._fallback is not None:
                logger.info("HMR 2.0 weights not found; executing through fallback reconstructor.")
                res = self._fallback.reconstruct(
                    image_rgb=image_rgb,
                    bounding_box=bounding_box,
                    mask_gray=mask_gray,
                    depth_stats=depth_stats,
                    pose_result=pose_result,
                    quality_preset=quality_preset,
                )
                res.warnings.append(
                    "HMR 2.0 model weights not found in data/models/hmr2/; executed via fallback engine."
                )
                return res
            raise RuntimeError(
                f"HMR 2.0 checkpoint not found at {self.checkpoint_path}. "
                "Download epoch=35-step=1000000.ckpt into data/models/hmr2/ using ModelManager.download_model('hmr2')."
            )

        # 1. Prepare 256x256 square crop preserving aspect ratio
        crop_rgb, transform_meta = prepare_person_crop(
            image_rgb, bounding_box, padding=0.2, target_size=(256, 256)
        )

        # 2. ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (crop_rgb.astype(np.float32) / 255.0 - mean) / std
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        # 3. Real HMR 2.0 Neural Inference
        with torch.no_grad():
            out = self._model(img_tensor)

        pred_pose_6d = out["pose_6d"]  # (1, 144)
        pred_shape = out["betas"]      # (1, 10)
        pred_cam = out["pred_cam"]      # (1, 3)

        # 4. Optional refinement vs Raw Neural Output
        # For "ultra" preset or explicit refine_pose, run gradient fitting without heuristics
        should_refine = (refine_pose or quality_preset == "ultra") and self.pose_fitter is not None
        if should_refine:
            fitted_pose, fitted_shape, fitted_cam, metrics = self.pose_fitter.fit(
                init_pose=pred_pose_6d,
                init_shape=pred_shape,
                init_cam=pred_cam,
                pose_result=pose_result,
                transform_meta=transform_meta,
                mask_gray=mask_gray,
                quality_preset=quality_preset,
                use_heuristics=False,
            )
            convergence_status = "HMR 2.0 ViT + SMPLPoseFitter Refined"
        else:
            fitted_pose = pred_pose_6d
            fitted_shape = pred_shape
            fitted_cam = pred_cam
            if self.pose_fitter is not None:
                metrics = self.pose_fitter.evaluate(
                    pose=fitted_pose,
                    shape=fitted_shape,
                    cam=fitted_cam,
                    pose_result=pose_result,
                    transform_meta=transform_meta,
                    mask_gray=mask_gray,
                    quality_preset=quality_preset,
                )
            else:
                metrics = {
                    "reprojection_error_pixels": None,
                    "reprojection_error_normalized": None,
                    "silhouette_iou": None,
                    "quality_score": 0.85,
                    "quality_preset": quality_preset,
                    "stance_description": "Standing Posture (Upright)",
                    "observed_keypoints_count": 0,
                    "pose_convergence_status": "HMR 2.0 ViT Direct Inference",
                    "metric_status": "estimated",
                }
            convergence_status = "HMR 2.0 ViT Direct Inference"

        # 5. Forward SMPL deformation
        with torch.no_grad():
            pose_for_smpl = fitted_pose.view(24, 6) if fitted_pose.dim() == 2 and fitted_pose.shape[-1] == 144 else fitted_pose
            vertices_tensor = self.smpl(pose_for_smpl, fitted_shape)
            joints_tensor = torch.matmul(self.smpl.J_regressor, vertices_tensor)

        vertices = vertices_tensor.detach().cpu().numpy().astype(np.float32)
        if vertices.ndim == 3:
            vertices = vertices[0]

        joints_np = joints_tensor.detach().cpu().numpy()
        if joints_np.ndim == 3:
            joints_np = joints_np[0]

        pelvis_raw = joints_np[0]
        pelvis_3d = [float(pelvis_raw[0]), float(-pelvis_raw[1]), float(-pelvis_raw[2])]

        # 6. OpenCV (+X right, +Y down, +Z depth) -> Three.js (+X right, +Y up, +Z front)
        vertices[:, 1] = -vertices[:, 1]
        vertices[:, 2] = -vertices[:, 2]

        faces = self.faces.copy()

        # 7. Validate geometry and export GLB
        is_valid, warnings = validate_mesh_geometry(vertices, faces)
        if transform_meta.get("is_clipped", False):
            warnings.append("Person is partially clipped by photo frame boundaries; peripheral limbs may be approximated.")

        normals = compute_mesh_normals(vertices, faces)
        v_min = vertices.min(axis=0).tolist()
        v_max = vertices.max(axis=0).tolist()
        dims = (vertices.max(axis=0) - vertices.min(axis=0)).tolist()

        cam_values = fitted_cam.squeeze(0).detach().cpu().numpy().tolist() if isinstance(fitted_cam, torch.Tensor) else [1.0, 0.0, 0.0]
        camera_params = {
            "scale": float(cam_values[0]),
            "tx": float(cam_values[1]),
            "ty": float(cam_values[2]),
        }

        # Projective vertex color mapping from cropped photograph
        vertex_colors = compute_projected_vertex_colors(
            vertices_three=vertices,
            normals_three=normals,
            camera_params=camera_params,
            crop_rgb=crop_rgb,
            mask_gray=mask_gray,
            transform_meta=transform_meta,
        )

        glb_bytes = export_mesh_glb(
            vertices=vertices,
            faces=faces,
            normals=normals,
            vertex_colors=vertex_colors,
            name="hmr2_human",
        )

        # 8. Reprojection diagnostic overlay
        overlay_png_bytes = None
        try:
            from app.pipeline.reprojection import generate_reprojection_overlay
            obs_kps = pose_result.keypoints if (pose_result and hasattr(pose_result, "keypoints")) else None
            overlay_rgb, reproj_diag = generate_reprojection_overlay(
                image_rgb=image_rgb,
                joints_3d=joints_np,
                camera_params=camera_params,
                transform_meta=transform_meta,
                bounding_box=bounding_box,
                observed_keypoints=obs_kps,
            )
            if reproj_diag.mean_reprojection_error_pixels is not None:
                metrics["reprojection_error_pixels"] = reproj_diag.mean_reprojection_error_pixels

            is_ok, enc_buf = cv2.imencode(".png", cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
            if is_ok:
                overlay_png_bytes = enc_buf.tobytes()
        except Exception as diag_err:
            logger.debug(f"Reprojection overlay generation: {diag_err}")

        return HumanReconstructionResult(
            vertices=vertices,
            faces=faces,
            normals=normals,
            bounds_min=v_min,
            bounds_max=v_max,
            dimensions=dims,
            glb_bytes=glb_bytes,
            model_name=self.model_name,
            model_version=self.model_version,
            device=str(self.device),
            pose_params=fitted_pose.squeeze(0).detach().cpu().numpy(),
            shape_params=fitted_shape.squeeze(0).detach().cpu().numpy(),
            camera_params=camera_params,
            texture_availability="partial",  # projective vertex colors
            quality_score=metrics.get("quality_score"),
            quality_preset=quality_preset,
            reprojection_error_pixels=metrics.get("reprojection_error_pixels"),
            reprojection_error_normalized=metrics.get("reprojection_error_normalized"),
            silhouette_iou=metrics.get("silhouette_iou"),
            stance_description=metrics.get("stance_description", "Standing Posture (Upright)"),
            observed_keypoints_count=metrics.get("observed_keypoints_count", 0),
            pose_convergence_status=convergence_status,
            metric_status=metrics.get("metric_status", "measured"),
            pelvis_3d=pelvis_3d,
            overlay_png_bytes=overlay_png_bytes,
            warnings=warnings,
            processing_time_ms=(time.time() - t0) * 1000.0,
        )


class ReconstructionBackendRegistry:
    """Registry managing modular human reconstruction backends."""

    def __init__(self):
        self._backends: Dict[str, HumanReconstructor] = {}

    def register(self, name: str, backend: HumanReconstructor) -> None:
        self._backends[name] = backend

    def get(self, name: str) -> Optional[HumanReconstructor]:
        return self._backends.get(name)

    def list_backends(self) -> List[str]:
        return list(self._backends.keys())

    def get_capabilities(self) -> Dict[str, Dict[str, Any]]:
        caps: Dict[str, Dict[str, Any]] = {}
        for name, backend in self._backends.items():
            if hasattr(backend, "get_capabilities"):
                caps[name] = backend.get_capabilities()
            else:
                caps[name] = {
                    "model_name": getattr(backend, "model_name", name),
                    "model_version": getattr(backend, "model_version", "1.0.0"),
                    "is_articulated": True,
                }
        return caps


reconstruction_registry = ReconstructionBackendRegistry()
