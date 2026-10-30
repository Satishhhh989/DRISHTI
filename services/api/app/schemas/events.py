"""DRISHTI - Processing Event Contract Models
Version 1.1.0 (Phase 02: Image Intelligence & Person Segmentation)
"""

from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field

ProcessingEventType = Literal[
    "job_accepted",
    "stage_started",
    "progress_updated",
    "intermediate_result_available",
    "asset_ready",
    "stage_completed",
    "stage_failed",
    "job_cancelled",
    "job_completed",
]

PipelineStageIdentifier = Literal[
    "image_analysis",
    "image_validation",
    "image_preparation",
    "person_detection",
    "detection_validation",
    "segmentation",
    "instance_segmentation",
    "mask_validation",
    "depth_estimation",
    "depth_normalization",
    "depth_association",
    "spatial_photo",
    "spatial_photo_preparation",
    "asset_persistence",
    "scene_update",
    "completed",
    "mesh_reconstruction",
    "human_reconstruction",
    "pose_estimation",
    "skeleton_fitting",
    "mesh_rigging",
    "animation_preparation",
    "scene_assembly",
]

JobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]

AssetCategoryType = Literal[
    "original_photo",
    "segmentation_mask",
    "depth_map",
    "depth_visualization",
    "mesh",
    "texture",
    "skeleton",
    "rig",
    "animation",
    "export",
]


class AssetReference(BaseModel):
    asset_id: str = Field(..., min_length=1)
    asset_type: AssetCategoryType
    uri: str = Field(..., min_length=1)
    mime_type: str = Field(..., min_length=1)
    size_bytes: Optional[int] = None


class StructuredError(BaseModel):
    code: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    stage: Optional[PipelineStageIdentifier] = None
    details: Optional[Dict[str, Any]] = None


class ProcessingEvent(BaseModel):
    event_id: str = Field(..., min_length=1, description="Unique event identifier (UUID)")
    job_id: str = Field(..., min_length=1, description="Target processing job identifier")
    scene_id: str = Field(..., min_length=1, description="Associated scene identifier")
    event_type: ProcessingEventType
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    sequence_number: int = Field(
        ..., ge=1, description="Strictly increasing integer per job"
    )
    stage_identifier: PipelineStageIdentifier
    status: JobStatus
    progress: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    message: Optional[str] = None
    entity_id: Optional[str] = None
    asset_reference: Optional[AssetReference] = None
    error: Optional[StructuredError] = None
