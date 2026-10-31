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

def _resolve_default_models_dir() -> Path:
    p_cwd = Path("./data/models").resolve()
    if p_cwd.exists():
        return p_cwd
    try:
        p_file = (Path(__file__).resolve().parents[4] / "data" / "models").resolve()
        if p_file.exists():
            return p_file
    except IndexError:
        pass
    return p_cwd

DEFAULT_MODELS_DIR = _resolve_default_models_dir()

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
    "yolov8m-seg": {
        "filename": "yolov8m-seg.pt",
        "category": "instance_segmentation",
        "description": "YOLOv8 Medium Instance Segmentation (High-precision masks & boundaries)",
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-seg.pt",
        "size_mb": 52.4,
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
    "spin-human-reconstruction": {
        "dir_name": "human_reconstruction",
        "filename": "human_reconstruction/spin_model_checkpoint.pth.tar",
        "category": "human_reconstruction",
        "description": "SPIN (SMPL Optimization in the Loop of Neural Networks) + SMPL Neutral LBS",
        "size_mb": 191.0,
        "license": "SPIN: MIT; SMPL: Non-Commercial Research License (MPI)",
        "task": "human-mesh-recovery",
    },
    "yolov8n-pose": {
        "filename": "yolov8n-pose.pt",
        "category": "pose_estimation",
        "description": "YOLOv8 Nano Human Pose Estimation (2D Keypoint Detection)",
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-pose.pt",
        "size_mb": 6.5,
        "license": "AGPL-3.0",
        "task": "pose",
    },
    "yolov8m-pose": {
        "filename": "yolov8m-pose.pt",
        "category": "pose_estimation",
        "description": "YOLOv8 Medium Human Pose Estimation (High-precision 2D Keypoint Detection)",
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m-pose.pt",
        "size_mb": 50.8,
        "license": "AGPL-3.0",
        "task": "pose",
    },
    "hmr2": {
        "dir_name": "hmr2",
        "filename": "hmr2/epoch=35-step=1000000.ckpt",
        "config_filename": "hmr2/model_config.yaml",
        "category": "human_reconstruction",
        "description": "4DHumans / HMR 2.0 (ViT Backbone + SMPL Transformer Regressor)",
        "hf_repo_id": "camenduru/4D-Humans",
        "hf_ckpt_filename": "HMR2/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt",
        "hf_cfg_filename": "HMR2/logs/train/multiruns/hmr2/0/model_config.yaml",
        "size_mb": 2584.0,
        "license": "Code: MIT; Checkpoint & SMPL: CC-BY-NC 4.0 (Non-Commercial Research)",
        "task": "human-mesh-recovery",
        "download_info": "Pretrained 4DHumans HMR 2.0 ViT checkpoint hosted on Hugging Face (camenduru/4D-Humans)",
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

        if model_key == "spin-human-reconstruction":
            from huggingface_hub import hf_hub_download
            target_dir = self.models_dir / meta["dir_name"]
            target_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading SPIN & SMPL assets to {target_dir}...")
            
            # 1. Download spin_model_checkpoint.pth.tar
            p1 = hf_hub_download(repo_id="camenduru/VIBE", filename="spin_model_checkpoint.pth.tar")
            dest_ckpt = target_dir / "spin_model_checkpoint.pth.tar"
            if not dest_ckpt.exists():
                import shutil
                shutil.copyfile(p1, dest_ckpt)

            # 2. Download smpl_mean_params.npz
            p2 = hf_hub_download(repo_id="camenduru/VIBE", filename="smpl_mean_params.npz")
            dest_mean = target_dir / "smpl_mean_params.npz"
            if not dest_mean.exists():
                import shutil
                shutil.copyfile(p2, dest_mean)

            # 3. Download SMPL_N_model.npz
            p3 = hf_hub_download(repo_id="lithiumice/models_hub", filename="4_SMPLhub/SMPL/X_model_npz/SMPL_N_model.npz")
            dest_smpl = target_dir / "SMPL_N_model.npz"
            if not dest_smpl.exists():
                import shutil
                shutil.copyfile(p3, dest_smpl)

            logger.info(f"SPIN model and SMPL assets saved to {target_dir}")
            return dest_ckpt

        if model_key == "hmr2":
            from huggingface_hub import hf_hub_download
            import shutil
            target_dir = self.models_dir / meta["dir_name"]
            target_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading HMR 2.0 assets from {meta['hf_repo_id']} to {target_dir}...")

            # 1. Download model_config.yaml
            dest_cfg = self.models_dir / meta["config_filename"]
            if not dest_cfg.exists():
                p_cfg = hf_hub_download(repo_id=meta["hf_repo_id"], filename=meta["hf_cfg_filename"])
                shutil.copyfile(p_cfg, dest_cfg)

            # 2. Download epoch checkpoint
            if not dest_path.exists() or dest_path.stat().st_size < 1000000:
                p_ckpt = hf_hub_download(repo_id=meta["hf_repo_id"], filename=meta["hf_ckpt_filename"])
                shutil.copyfile(p_ckpt, dest_path)

            logger.info(f"HMR 2.0 checkpoint and config saved to {dest_path}")
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
