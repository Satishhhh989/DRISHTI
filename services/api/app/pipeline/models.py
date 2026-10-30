"""DRISHTI - Model Management & Weight Discovery

Manages local model checkpoints under data/models/, verifies file presence,
and provides explicit weight downloading.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import urllib.request

from app.core.config import settings
from app.core.logging import logger

DEFAULT_MODELS_DIR = Path("./data/models").resolve()

# Pretrained model catalog for Phase 02 and Phase 03
MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "yolov8n-seg": {
        "filename": "yolov8n-seg.pt",
        "category": "instance_segmentation",
        "description": "YOLOv8 Nano Instance Segmentation (Multi-person detection + pixel masks)",
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt",
        "size_mb": 6.7,
        "license": "AGPL-3.0",
        "task": "segment",
    },
    "yolov8s-seg": {
        "filename": "yolov8s-seg.pt",
        "category": "instance_segmentation",
        "description": "YOLOv8 Small Instance Segmentation (Higher accuracy person masks)",
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-seg.pt",
        "size_mb": 22.8,
        "license": "AGPL-3.0",
        "task": "segment",
    },
    "depth-anything-v2-small": {
        "dir_name": "depth-anything-v2-small",
        "filename": "depth-anything-v2-small/model.safetensors",
        "category": "depth_estimation",
        "description": "Depth Anything V2 Small (Monocular relative scene depth)",
        "hf_model_id": "depth-anything/Depth-Anything-V2-Small-hf",
        "size_mb": 99.2,
        "license": "Apache-2.0",
        "task": "depth-estimation",
    },
}


class ModelManager:
    """Manages model checkpoints, validation, and lifecycle."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = (models_dir or DEFAULT_MODELS_DIR).resolve()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_models: Dict[str, Any] = {}

    def get_model_path(self, model_key: str = "yolov8n-seg") -> Path:
        meta = MODEL_CATALOG.get(model_key)
        if not meta:
            raise KeyError(f"Unknown model identifier: '{model_key}'")
        return self.models_dir / meta["filename"]

    def is_model_available(self, model_key: str = "yolov8n-seg") -> bool:
        path = self.get_model_path(model_key)
        return path.exists() and path.stat().st_size > 1000

    def get_model_status(self, model_key: str = "yolov8n-seg") -> Dict[str, Any]:
        meta = MODEL_CATALOG.get(model_key)
        if not meta:
            raise KeyError(f"Unknown model identifier: '{model_key}'")

        path = self.get_model_path(model_key)
        available = path.exists() and path.stat().st_size > 1000
        size_bytes = path.stat().st_size if available else 0

        return {
            "model_key": model_key,
            "filename": meta["filename"],
            "description": meta["description"],
            "size_mb": meta["size_mb"],
            "license": meta["license"],
            "available": available,
            "path": str(path) if available else None,
            "actual_size_bytes": size_bytes,
        }

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        return {k: self.get_model_status(k) for k in MODEL_CATALOG}

    def download_model(self, model_key: str = "yolov8n-seg") -> Path:
        """Explicitly downloads model weights to data/models/."""
        meta = MODEL_CATALOG.get(model_key)
        if not meta:
            raise KeyError(f"Unknown model identifier: '{model_key}'")

        dest_path = self.get_model_path(model_key)
        if dest_path.exists() and dest_path.stat().st_size > 1000:
            logger.info(f"Model '{model_key}' already cached at {dest_path}")
            return dest_path

        if "hf_model_id" in meta:
            hf_id = meta["hf_model_id"]
            target_dir = self.models_dir / meta["dir_name"]
            logger.info(f"Downloading Hugging Face model '{model_key}' from {hf_id} to {target_dir}...")
            try:
                from transformers import AutoImageProcessor, AutoModelForDepthEstimation
                target_dir.mkdir(parents=True, exist_ok=True)
                processor = AutoImageProcessor.from_pretrained(hf_id)
                model = AutoModelForDepthEstimation.from_pretrained(hf_id)
                processor.save_pretrained(target_dir)
                model.save_pretrained(target_dir)
                logger.info(f"Model '{model_key}' saved successfully to {target_dir}")
                return dest_path
            except Exception as e:
                logger.error(f"Failed to download Hugging Face model '{model_key}': {e}")
                raise

        url = meta["download_url"]
        logger.info(f"Downloading model '{model_key}' ({meta['size_mb']} MB) from {url} to {dest_path}...")
        
        tmp_dest = dest_path.with_suffix(".tmp")
        try:
            urllib.request.urlretrieve(url, tmp_dest)
            tmp_dest.rename(dest_path)
            logger.info(f"Model '{model_key}' downloaded successfully: {dest_path}")
            return dest_path
        except Exception as e:
            if tmp_dest.exists():
                tmp_dest.unlink()
            logger.error(f"Failed to download model weights for '{model_key}': {e}")
            raise


model_manager = ModelManager()
