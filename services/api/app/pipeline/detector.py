"""DRISHTI - Real Person Detection Engine (YOLOv8)

Runs local person detection, generates normalized [0, 1] and pixel coordinates,
assigns stable entity IDs, and captures model confidence.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
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
        return self.detect_entities(
            image_path=image_path,
            confidence_threshold=confidence_threshold,
            detect_objects=False,
        )

    def detect_entities(
        self,
        image_path: Union[Path, str],
        confidence_threshold: float = 0.35,
        detect_objects: bool = True,
    ) -> List[DetectionResult]:
        """Runs multi-entity detection (persons and salient objects) using YOLOv8."""
        image_path = Path(image_path)
        model = self._get_or_load_model()

        # Read dimensions to normalize coordinates accurately
        with Image.open(image_path) as img:
            orig_w, orig_h = img.size

        # Run inference on the verified device
        device = device_manager.device
        logger.info(
            f"Running YOLO detection on '{image_path.name}' ({orig_w}x{orig_h}) using device='{device}' (conf={confidence_threshold}, detect_objects={detect_objects})"
        )

        predict_kwargs: dict = {
            "source": str(image_path),
            "conf": confidence_threshold,
            "device": device,
            "verbose": False,
        }
        if not detect_objects:
            predict_kwargs["classes"] = [0]  # Only person

        results = model.predict(**predict_kwargs)

        detections: List[DetectionResult] = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        class_names = results[0].names or {}

        for idx in range(len(boxes)):
            box = boxes[idx]
            cls_id = int(box.cls[0])
            class_name = class_names.get(cls_id, f"class_{cls_id}").lower()
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

            # Minimum area filter for small noisy objects (must be at least 0.1% of image)
            box_area_ratio = ((x2 - x1) * (y2 - y1)) / float(orig_w * orig_h)
            if cls_id != 0 and box_area_ratio < 0.001:
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

            is_person = (cls_id == 0)
            entity_type = "person" if is_person else "object"
            entity_id = f"person-{uuid.uuid4().hex[:8]}" if is_person else f"object-{uuid.uuid4().hex[:8]}"
            label = "person" if is_person else class_name

            detections.append(
                DetectionResult(
                    bounding_box=norm_box,
                    pixel_box=(round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)),
                    confidence=round(conf, 4),
                    entity_id=entity_id,
                    label=label,
                    model=self.model_key,
                    entity_type=entity_type,
                    object_class=None if is_person else class_name,
                )
            )

        people_count = sum(1 for d in detections if d.entity_type == "person")
        object_count = sum(1 for d in detections if d.entity_type == "object")
        logger.info(f"YOLO detector identified {people_count} people and {object_count} objects in '{image_path.name}'")
        return detections
