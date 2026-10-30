"""PHOTO → 3D ENGINE - Modular AI Inference Interfaces

Defines the pluggable contracts for Phase 02 through Phase 10 AI inference stages.
No heavy AI models or weights are loaded in Phase 01.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from app.schemas.assets import AssetMetadata
from app.schemas.common import BoundingBox2D
from app.schemas.scene import SceneEntity


class DetectionResult:
    def __init__(
        self,
        bounding_box: BoundingBox2D,
        confidence: float,
        label: str = "person"
    ):
        self.bounding_box = bounding_box
        self.confidence = confidence
        self.label = label


class IPersonDetector(ABC):
    """Interface for multi-person detection (Phase 02)."""

    @abstractmethod
    async def detect_people(self, photo_asset: AssetMetadata) -> List[DetectionResult]:
        """Detects all human subjects within the source photograph."""
        pass


class ISegmentor(ABC):
    """Interface for person instance segmentation and background matting (Phase 03)."""

    @abstractmethod
    async def segment_person(
        self,
        photo_asset: AssetMetadata,
        detection: DetectionResult
    ) -> AssetMetadata:
        """Produces a high-resolution alpha segmentation mask asset."""
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
