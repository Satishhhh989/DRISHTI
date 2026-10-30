"""DRISHTI - Real Person Instance Segmentation Engine (YOLOv8-seg)

Extracts individual human alpha masks, aligns coordinates with the original photograph,
and produces lossless 8-bit binary PNG mask assets.
"""

import io
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image

from app.core.logging import logger
from app.pipeline.device import device_manager
from app.pipeline.interfaces import DetectionResult, ISegmentor, SegmentationResult
from app.pipeline.models import model_manager

try:
    from ultralytics import YOLO
    import torch
    import torch.nn.functional as F
    TORCH_AND_YOLO_AVAILABLE = True
except ImportError:
    TORCH_AND_YOLO_AVAILABLE = False


class YoloPersonSegmentor(ISegmentor):
    """Local instance segmentation engine generating per-person masks."""

    def __init__(self, model_key: str = "yolov8n-seg"):
        self.model_key = model_key
        self._model: Optional[YOLO] = None

    def _get_or_load_model(self) -> YOLO:
        if self._model is not None:
            return self._model

        if not TORCH_AND_YOLO_AVAILABLE:
            raise RuntimeError("PyTorch or Ultralytics is not available in the environment.")

        model_path = model_manager.get_model_path(self.model_key)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model weights for '{self.model_key}' not found at {model_path}."
            )

        self._model = YOLO(str(model_path))
        return self._model

    def segment_people(
        self,
        image_path: Path,
        detections: List[DetectionResult],
    ) -> List[SegmentationResult]:
        """Generates individual binary PNG masks matching the original image dimensions."""
        if not detections:
            return []

        model = self._get_or_load_model()

        with Image.open(image_path) as img:
            orig_w, orig_h = img.size

        device = device_manager.device
        logger.info(
            f"Running YOLO instance segmentation on '{image_path.name}' ({orig_w}x{orig_h}) for {len(detections)} people"
        )

        results = model.predict(
            source=str(image_path),
            classes=[0],
            device=device,
            verbose=False,
        )

        if not results or len(results) == 0 or results[0].masks is None:
            logger.warning(f"No mask tensors produced for '{image_path.name}'")
            return []

        mask_data = results[0].masks.data  # Tensor of shape [N, mask_h, mask_w]
        n_masks = mask_data.shape[0]

        segmentation_results: List[SegmentationResult] = []

        # Iterate over detections and associate masks
        for i, det in enumerate(detections):
            if i >= n_masks:
                logger.warning(f"Detection {i} has no corresponding mask tensor (total masks={n_masks})")
                continue

            try:
                # 1. Extract single mask tensor [1, 1, mask_h, mask_w]
                single_mask = mask_data[i : i + 1].unsqueeze(0).float()

                # 2. Upsample to original image resolution (orig_h, orig_w)
                upsampled = F.interpolate(
                    single_mask,
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                )

                # 3. Convert to binary numpy array [orig_h, orig_w] (threshold at 0.5)
                mask_np = (upsampled[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
                pixel_count = int(np.count_nonzero(mask_np))

                # 4. Encode as 8-bit grayscale PNG in memory
                mask_img = Image.fromarray(mask_np, mode="L")
                buf = io.BytesIO()
                mask_img.save(buf, format="PNG", optimize=True)
                png_bytes = buf.getvalue()

                segmentation_results.append(
                    SegmentationResult(
                        entity_id=det.entity_id,
                        mask_png_bytes=png_bytes,
                        confidence=det.confidence,
                        pixel_count=pixel_count,
                        width=orig_w,
                        height=orig_h,
                        model=self.model_key,
                    )
                )
            except Exception as e:
                logger.error(f"Failed to extract mask for entity '{det.entity_id}': {e}")

        logger.info(
            f"Successfully produced {len(segmentation_results)} segmentation masks for '{image_path.name}'"
        )
        return segmentation_results
