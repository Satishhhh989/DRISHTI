"""DRISHTI - Real Person Instance Segmentation Engine (YOLOv8-seg)

Extracts individual human alpha masks, aligns coordinates with the original photograph,
and produces lossless 8-bit binary PNG mask assets.
"""

import io
from pathlib import Path
from typing import List, Optional, Tuple, Union
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
        image_path: Union[Path, str],
        detections: List[DetectionResult],
    ) -> List[SegmentationResult]:
        """Generates individual binary PNG masks matching the original image dimensions."""
        return self.segment_entities(image_path=image_path, detections=detections)

    def segment_entities(
        self,
        image_path: Union[Path, str],
        detections: List[DetectionResult],
    ) -> List[SegmentationResult]:
        """Generates individual binary PNG masks for all detected entities (people and objects)."""
        if not detections:
            return []

        image_path = Path(image_path)
        model = self._get_or_load_model()

        with Image.open(image_path) as img:
            orig_w, orig_h = img.size

        device = device_manager.device
        logger.info(
            f"Running YOLO instance segmentation on '{image_path.name}' ({orig_w}x{orig_h}) for {len(detections)} entities"
        )

        results = model.predict(
            source=str(image_path),
            device=device,
            verbose=False,
        )

        if not results or len(results) == 0 or results[0].masks is None:
            logger.warning(f"No mask tensors produced for '{image_path.name}'")
            return []

        mask_data = results[0].masks.data  # Tensor of shape [N, mask_h, mask_w]
        n_masks = mask_data.shape[0]

        pred_boxes: List[Tuple[float, float, float, float]] = []
        if results[0].boxes is not None:
            for b in results[0].boxes:
                xyxy = b.xyxy[0].tolist()
                pred_boxes.append((float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])))

        def _calc_iou(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
            xl = max(b1[0], b2[0])
            yt = max(b1[1], b2[1])
            xr = min(b1[2], b2[2])
            yb = min(b1[3], b2[3])
            if xr <= xl or yb <= yt:
                return 0.0
            inter = (xr - xl) * (yb - yt)
            a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            union = a1 + a2 - inter
            return inter / union if union > 0 else 0.0

        segmentation_results: List[SegmentationResult] = []

        # Iterate over detections and associate masks via IoU matching
        for i, det in enumerate(detections):
            best_mask_idx = i if i < n_masks else -1
            best_iou = -1.0

            if pred_boxes:
                for p_idx, p_box in enumerate(pred_boxes):
                    if p_idx < n_masks:
                        iou = _calc_iou(det.pixel_box, p_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_mask_idx = p_idx

            if best_mask_idx < 0 or best_mask_idx >= n_masks:
                logger.warning(f"Detection {det.entity_id} has no corresponding mask tensor (total masks={n_masks})")
                continue

            try:
                # 1. Extract single mask tensor [1, 1, mask_h, mask_w]
                single_mask = mask_data[best_mask_idx : best_mask_idx + 1].unsqueeze(0).float()

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
