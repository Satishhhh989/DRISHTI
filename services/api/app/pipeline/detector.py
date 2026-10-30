"""DRISHTI - Real Person Detection Engine (YOLOv8)

Runs local person detection, generates normalized [0, 1] and pixel coordinates,
assigns stable entity IDs, and captures model confidence.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import uuid
from PIL import Image

from app.core.logging import logger
from app.pipeline.device import device_manager
from app.pipeline.interfaces import DetectionResult, IPersonDetector
from app.pipeline.models import model_manager
from app.schemas.common import BoundingBox2D

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class YoloPersonDetector(IPersonDetector):
    """Local multi-person detector using pretrained YOLOv8 weights."""

    def __init__(self, model_key: str = "yolov8n-seg"):
        self.model_key = model_key
        self._model: Optional[YOLO] = None

    def _get_or_load_model(self) -> YOLO:
        if self._model is not None:
            return self._model

        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("Ultralytics library is not installed in the environment.")

        model_path = model_manager.get_model_path(self.model_key)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model weights for '{self.model_key}' not found at {model_path}. Please download weights first."
            )

        logger.info(f"Loading YOLO detector model from {model_path} onto {device_manager.device}...")
        self._model = YOLO(str(model_path))
        return self._model

    def detect_people(
        self,
        image_path: Path,
        confidence_threshold: float = 0.35,
    ) -> List[DetectionResult]:
        """Runs person detection (class 0 in COCO) and returns normalized and pixel coordinates."""
        model = self._get_or_load_model()

        # Read dimensions to normalize coordinates accurately
        with Image.open(image_path) as img:
            orig_w, orig_h = img.size

        # Run inference on the verified device
        device = device_manager.device
        logger.info(
            f"Running YOLO person detection on '{image_path.name}' ({orig_w}x{orig_h}) using device='{device}' (conf={confidence_threshold})"
        )

        results = model.predict(
            source=str(image_path),
            classes=[0],  # COCO class 0: person
            conf=confidence_threshold,
            device=device,
            verbose=False,
        )

        detections: List[DetectionResult] = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for idx in range(len(boxes)):
            box = boxes[idx]
            xyxy: List[float] = box.xyxy[0].tolist()  # [x1, y1, x2, y2] in pixels
            conf = float(box.conf[0])

            # Clamp coordinates to original image bounds
            x1 = max(0.0, min(float(orig_w), xyxy[0]))
            y1 = max(0.0, min(float(orig_h), xyxy[1]))
            x2 = max(0.0, min(float(orig_w), xyxy[2]))
            y2 = max(0.0, min(float(orig_h), xyxy[3]))

            # Validate non-degenerate box
            if x2 <= x1 or y2 <= y1:
                continue

            # Normalized coordinates in [0, 1]
            x_min = round(x1 / orig_w, 6)
            y_min = round(y1 / orig_h, 6)
            x_max = round(x2 / orig_w, 6)
            y_max = round(y2 / orig_h, 6)

            # Ensure strict min < max
            x_max = max(x_min + 0.001, min(1.0, x_max))
            y_max = max(y_min + 0.001, min(1.0, y_max))

            norm_box = BoundingBox2D(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            )

            # Generate stable entity ID
            entity_id = f"person-{uuid.uuid4().hex[:8]}"

            detections.append(
                DetectionResult(
                    bounding_box=norm_box,
                    pixel_box=(round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)),
                    confidence=round(conf, 4),
                    entity_id=entity_id,
                    label="person",
                    model=self.model_key,
                )
            )

        logger.info(f"YOLO detector identified {len(detections)} people in '{image_path.name}'")
        return detections
