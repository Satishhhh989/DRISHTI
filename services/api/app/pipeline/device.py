"""DRISHTI - Inference Device Manager

Detects, verifies, and selects the optimal local compute device:
Apple Silicon (MPS), NVIDIA CUDA, or CPU fallback.
"""

from typing import Dict, Any, Optional
import platform
from app.core.logging import logger

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DeviceManager:
    """Manages inference device selection and capability checks."""

    def __init__(self, preferred_device: Optional[str] = None):
        self._preferred = preferred_device
        self._detected_device: Optional[str] = None
        self._device_name: str = "Unknown"
        self._verified: bool = False
        self._init_device()

    def _init_device(self) -> None:
        if not TORCH_AVAILABLE:
            self._detected_device = "cpu"
            self._device_name = "CPU (PyTorch not available)"
            self._verified = True
            return

        # 1. Manual override if specified
        if self._preferred in ("mps", "cuda", "cpu"):
            if self._preferred == "cuda" and torch.cuda.is_available():
                self._detected_device = "cuda"
                self._device_name = torch.cuda.get_device_name(0)
            elif self._preferred == "mps" and torch.backends.mps.is_available():
                self._detected_device = "mps"
                self._device_name = f"Apple Silicon ({platform.processor() or 'ARM'})"
            else:
                self._detected_device = "cpu"
                self._device_name = f"CPU ({platform.machine()})"
        else:
            # 2. Automatic selection: CUDA > MPS > CPU
            if torch.cuda.is_available():
                self._detected_device = "cuda"
                self._device_name = torch.cuda.get_device_name(0)
            elif torch.backends.mps.is_available():
                self._detected_device = "mps"
                self._device_name = f"Apple Silicon Metal Performance Shaders ({platform.processor() or 'ARM'})"
            else:
                self._detected_device = "cpu"
                self._device_name = f"CPU ({platform.machine()})"

        # 3. Verify tensor operations on the selected device
        self._verify_device()

    def _verify_device(self) -> None:
        if not TORCH_AVAILABLE:
            self._verified = True
            return

        try:
            device_str = self._detected_device or "cpu"
            test_tensor = torch.ones((2, 2), device=device_str)
            _ = test_tensor + 1.0
            self._verified = True
            logger.info(
                f"Inference device verified: '{device_str}' ({self._device_name})"
            )
        except Exception as e:
            logger.warning(
                f"Device '{self._detected_device}' failed verification: {e}. Falling back to CPU."
            )
            self._detected_device = "cpu"
            self._device_name = f"CPU fallback ({platform.machine()})"
            self._verified = True

    @property
    def device(self) -> str:
        return self._detected_device or "cpu"

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def is_accelerated(self) -> bool:
        return self.device in ("mps", "cuda")

    def get_device_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "device": self.device,
            "device_name": self.device_name,
            "is_accelerated": self.is_accelerated,
            "verified": self._verified,
            "torch_version": torch.__version__ if TORCH_AVAILABLE else None,
            "platform": platform.platform(),
            "processor": platform.processor(),
        }
        if TORCH_AVAILABLE:
            info["mps_available"] = bool(torch.backends.mps.is_available())
            info["cuda_available"] = bool(torch.cuda.is_available())
        return info


device_manager = DeviceManager()
