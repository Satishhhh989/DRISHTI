from app.pipeline.interfaces import (
    IPersonDetector,
    ISegmentor,
    IDepthEstimator,
    IMeshReconstructor,
    ISkeletonFitter,
    IAnimator,
    DetectionResult,
    SegmentationResult,
)
from app.pipeline.device import DeviceManager, device_manager
from app.pipeline.models import ModelManager, model_manager
from app.pipeline.detector import YoloPersonDetector
from app.pipeline.segmentor import YoloPersonSegmentor
from app.pipeline.mock import MockPersonDetector, MockPersonSegmentor
from app.pipeline.coordinator import PipelineCoordinator, pipeline_coordinator

__all__ = [
    "IPersonDetector",
    "ISegmentor",
    "IDepthEstimator",
    "IMeshReconstructor",
    "ISkeletonFitter",
    "IAnimator",
    "DetectionResult",
    "SegmentationResult",
    "DeviceManager",
    "device_manager",
    "ModelManager",
    "model_manager",
    "YoloPersonDetector",
    "YoloPersonSegmentor",
    "MockPersonDetector",
    "MockPersonSegmentor",
    "PipelineCoordinator",
    "pipeline_coordinator",
]
