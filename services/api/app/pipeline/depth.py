"""DRISHTI - Depth Estimation & Normalization Engine
Phase 03: Monocular Relative Depth Estimation using Depth Anything V2.

Canonical Depth Convention:
- Canonical Proximity Metric P in [0.0, 1.0]
- P = 1.0 represents the nearest foreground surface (closest to camera)
- P = 0.0 represents the farthest background surface (furthest from camera)
- Depth Anything V2 outputs distance-like relative depth (smaller raw value = closer),
  which is converted during normalization to the canonical proximity convention.
"""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from app.core.logging import logger
from app.pipeline.device import device_manager
from app.pipeline.interfaces import DepthEstimationResult, IDepthEstimator
from app.pipeline.models import model_manager
from app.schemas.scene import EntityDepthInfo


def normalize_depth_map(
    raw_depth: np.ndarray,
    model_convention: str = "distance_smaller_is_closer",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Normalizes raw depth output into canonical relative proximity [0, 1].

    Uses robust percentile clamping (2nd to 98th percentile) to protect against
    extreme outliers and specular artifacts. Handles pathological inputs
    including NaNs, infinities, and constant depth fields.
    """
    if raw_depth.size == 0:
        raise ValueError("Cannot normalize empty depth map")

    # Replace NaNs and Infinities safely
    clean = np.nan_to_num(raw_depth, nan=0.0, posinf=0.0, neginf=0.0)
    valid_mask = np.isfinite(raw_depth) & (raw_depth != 0.0)
    valid_vals = raw_depth[valid_mask] if np.any(valid_mask) else clean.flatten()

    if len(valid_vals) == 0:
        logger.warning("Depth map contains no valid non-zero values; defaulting to flat field")
        normalized = np.full_like(clean, 0.5, dtype=np.float32)
        meta = {
            "p_min": 0.0,
            "p_max": 1.0,
            "raw_min": 0.0,
            "raw_max": 1.0,
            "model_convention": model_convention,
            "canonical_convention": "relative_proximity",
            "method": "percentile_clamped",
        }
        return normalized, meta

    raw_min = float(valid_vals.min())
    raw_max = float(valid_vals.max())
    p_low = float(np.percentile(valid_vals, 2))
    p_high = float(np.percentile(valid_vals, 98))

    # Guard against constant or near-constant depth maps
    if abs(p_high - p_low) < 1e-6:
        logger.debug("Constant depth field detected; assigning neutral proximity 0.5")
        normalized = np.full_like(clean, 0.5, dtype=np.float32)
    else:
        clamped = np.clip(clean, p_low, p_high)
        if model_convention == "distance_smaller_is_closer":
            # Smaller raw value = closer -> proximity is inverted
            normalized = (p_high - clamped) / (p_high - p_low)
        else:
            # Larger raw value = closer (disparity convention)
            normalized = (clamped - p_low) / (p_high - p_low)
        normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)

    meta = {
        "p_min": round(p_low, 4),
        "p_max": round(p_high, 4),
        "raw_min": round(raw_min, 4),
        "raw_max": round(raw_max, 4),
        "model_convention": model_convention,
        "canonical_convention": "relative_proximity",
        "method": "percentile_clamped",
    }
    return normalized, meta


def generate_depth_visualization(normalized_proximity: np.ndarray) -> bytes:
    """Generates an 8-bit grayscale PNG inspection visualization.

    Bright pixels (255) represent closer foreground surfaces;
    dark pixels (0) represent farther background surfaces.
    """
    clipped = np.clip(normalized_proximity, 0.0, 1.0)
    u8 = (clipped * 255.0).astype(np.uint8)
    img = Image.fromarray(u8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def compute_entity_depth_stats(
    normalized_proximity: np.ndarray,
    mask_bytes: Optional[bytes] = None,
    pixel_box: Optional[List[float]] = None,
    entity_id: str = "unknown",
) -> EntityDepthInfo:
    """Computes robust depth statistics for a person entity.

    Prioritizes the person's instance segmentation mask. If unavailable, falls
    back to the 2D bounding box with explicit documentation.
    """
    h, w = normalized_proximity.shape
    fg_depths: Optional[np.ndarray] = None
    method: str = "segmentation_mask"

    # Attempt 1: Segmentation mask
    if mask_bytes and len(mask_bytes) > 0:
        try:
            with Image.open(io.BytesIO(mask_bytes)) as mask_img:
                mask_arr = np.array(mask_img.convert("L"))
                if mask_arr.shape != (h, w):
                    # Resize mask to match depth map dimensions if necessary
                    resized_mask = mask_img.resize((w, h), Image.Resampling.NEAREST)
                    mask_arr = np.array(resized_mask.convert("L"))
                
                fg_mask = mask_arr > 128
                if np.any(fg_mask):
                    fg_depths = normalized_proximity[fg_mask]
                    method = "segmentation_mask"
        except Exception as e:
            logger.warning(f"Could not extract mask depth for entity {entity_id}: {e}")

    # Attempt 2: Bounding box fallback
    if (fg_depths is None or len(fg_depths) == 0) and pixel_box and len(pixel_box) == 4:
        x1 = max(0, min(w - 1, int(pixel_box[0])))
        y1 = max(0, min(h - 1, int(pixel_box[1])))
        x2 = max(x1 + 1, min(w, int(pixel_box[2])))
        y2 = max(y1 + 1, min(h, int(pixel_box[3])))
        fg_depths = normalized_proximity[y1:y2, x1:x2].flatten()
        method = "bounding_box"

    if fg_depths is None or len(fg_depths) == 0:
        return EntityDepthInfo(
            status="failed",
            median_proximity=0.5,
            valid_pixel_coverage=0.0,
            method="bounding_box",
            error="No valid pixels found for depth calculation",
        )

    # Compute robust statistics
    median_val = float(np.median(fg_depths))
    p5 = float(np.percentile(fg_depths, 5))
    p95 = float(np.percentile(fg_depths, 95))
    variation = float(np.std(fg_depths))
    coverage = float(len(fg_depths) / (h * w))

    return EntityDepthInfo(
        status="ready",
        median_proximity=round(median_val, 4),
        min_proximity=round(p5, 4),
        max_proximity=round(p95, 4),
        depth_variation=round(variation, 4),
        valid_pixel_coverage=round(coverage, 4),
        method=method,  # type: ignore[arg-type]
    )


class DepthAnythingV2Estimator(IDepthEstimator):
    """Local Monocular Depth Estimator using Depth Anything V2 Small."""

    def __init__(self, model_key: str = "depth-anything-v2-small"):
        self.model_key = model_key
        self._processor: Any = None
        self._model: Any = None
        self._device: Optional[torch.device] = None

    def _load_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        model_dir = model_manager.models_dir / "depth-anything-v2-small"
        if not (model_dir / "model.safetensors").exists():
            raise FileNotFoundError(
                f"Depth Anything V2 weights not found at {model_dir}. Please run model download first."
            )

        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._device = device_manager.get_torch_device()
        logger.info(f"Loading Depth Anything V2 from {model_dir} onto {self._device}...")
        self._processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
        self._model = AutoModelForDepthEstimation.from_pretrained(str(model_dir), local_files_only=True)
        self._model = self._model.to(self._device)
        self._model.eval()
        logger.info(f"Depth Anything V2 successfully initialized on {self._device}")

    def estimate_depth(self, image_path: Path) -> DepthEstimationResult:
        self._load_model()

        if not image_path.exists():
            raise FileNotFoundError(f"Source image not found: {image_path}")

        with Image.open(image_path) as pil_img:
            rgb_img = pil_img.convert("RGB")
            orig_w, orig_h = rgb_img.size

        # Preprocess and forward pass
        inputs = self._processor(images=rgb_img, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            predicted_depth = outputs.predicted_depth  # Shape: [1, H_model, W_model]

            # Bilinearly interpolate to exact original image dimensions
            interpolated = F.interpolate(
                predicted_depth.unsqueeze(1),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        raw_depth = interpolated.squeeze(0).cpu().numpy().astype(np.float32)

        # Normalize depth using canonical proximity convention
        normalized, norm_meta = normalize_depth_map(
            raw_depth,
            model_convention="distance_smaller_is_closer",
        )

        # Generate inspection visualization PNG
        vis_bytes = generate_depth_visualization(normalized)

        return DepthEstimationResult(
            raw_depth=raw_depth,
            normalized_depth=normalized,
            visualization_png_bytes=vis_bytes,
            width=orig_w,
            height=orig_h,
            model_name="depth-anything-v2-small",
            device=str(self._device),
            normalization_metadata=norm_meta,
            convention="relative_proximity",
        )

