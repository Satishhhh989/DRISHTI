"""DRISHTI - Modular AI Inference Interfaces

Defines the pluggable contracts for Phase 02 through Phase 10 AI inference stages.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple
from app.schemas.assets import AssetMetadata
from app.schemas.common import BoundingBox2D
from app.schemas.scene import SceneEntity


class DetectionResult:
    """Standardized multi-person detection result."""

    def __init__(
        self,
        bounding_box: BoundingBox2D,
        pixel_box: Tuple[float, float, float, float],
        confidence: float,
        entity_id: str,
        label: str = "person",
        model: str = "yolov8n-seg",
    ):
        self.bounding_box = bounding_box
        self.pixel_box = pixel_box
        self.confidence = confidence
        self.entity_id = entity_id
        self.label = label
        self.model = model


class SegmentationResult:
    """Standardized individual person segmentation mask result."""

    def __init__(
        self,
        entity_id: str,
        mask_png_bytes: bytes,
        confidence: float,
        pixel_count: int,
        width: int,
        height: int,
        model: str = "yolov8n-seg",
    ):
        self.entity_id = entity_id
        self.mask_png_bytes = mask_png_bytes
        self.confidence = confidence
        self.pixel_count = pixel_count
        self.width = width
        self.height = height
        self.model = model


class IPersonDetector(ABC):
    """Interface for multi-person detection (Phase 02)."""

    @abstractmethod
    def detect_people(
        self,
        image_path: Path,
        confidence_threshold: float = 0.35,
    ) -> List[DetectionResult]:
        """Detects all human subjects within the source photograph."""
        pass


class ISegmentor(ABC):
    """Interface for person instance segmentation (Phase 02/03)."""

    @abstractmethod
    def segment_people(
        self,
        image_path: Path,
        detections: List[DetectionResult],
    ) -> List[SegmentationResult]:
        """Produces individual binary alpha masks for each detected person."""
        pass


class IDepthEstimator(ABC):
    """Interface for monocular scene depth estimation (Phase 04)."""

    @abstractmethod
    async def estimate_depth(self, photo_asset: AssetMetadata) -> AssetMetadata:
        """Estimates scene depth map asset."""
        pass


class IMeshReconstructor(ABC):
    """Interface for 3D human mesh reconstruction (Phase 06)."""

    @abstractmethod
    async def reconstruct_mesh(
        self,
        photo_asset: AssetMetadata,
        mask_asset: AssetMetadata,
        entity: SceneEntity
    ) -> AssetMetadata:
        """Reconstructs 3D polygon mesh geometry (OBJ/GLTF) for a detected person."""
        pass


class ISkeletonFitter(ABC):
    """Interface for 3D human pose and skeleton fitting (Phase 08)."""

    @abstractmethod
    async def fit_skeleton(
        self,
        mesh_asset: AssetMetadata,
        entity: SceneEntity
    ) -> AssetMetadata:
        """Fits a standard humanoid kinematic skeleton and bone weights."""
        pass


class IAnimator(ABC):
    """Interface for independent character animation (Phase 09)."""

    @abstractmethod
    async def apply_animation(
        self,
        skeleton_asset: AssetMetadata,
        clip_name: str
    ) -> None:
        """Applies motion data to reconstructed humanoid rig."""
        pass
