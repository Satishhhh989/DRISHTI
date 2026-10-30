"""DRISHTI - Scene Schema Models
Version 1.1.0 (Phase 02: Image Intelligence & Person Segmentation)
"""

from datetime import datetime, timezone
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
from app.schemas.common import (
    BoundingBox2D,
    CoordinateSystemInfo,
    Transform,
)

SCHEMA_VERSION = "1.0.0"

EntityType = Literal["person", "background", "environment", "prop"]
ReconstructionStatus = Literal["unstarted", "detecting", "reconstructing", "ready", "failed"]
RepresentationStatus = Literal["unavailable", "pending", "processing", "ready", "failed"]
SceneProcessingStatus = Literal["unprocessed", "processing", "ready", "error"]


class ConfidenceMetadata(BaseModel):
    detection_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    segmentation_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reconstruction_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    notes: Optional[str] = None


class AnimationState(BaseModel):
    current_clip: Optional[str] = None
    is_playing: bool = False
    playback_speed: float = 1.0
    loop: bool = True


class DetectionInfo(BaseModel):
    confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: BoundingBox2D
    pixel_box: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x_min, y_min, x_max, y_max] in image pixel coordinates"
    )
    model: str
    detected_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: Literal["ready", "failed"] = "ready"


class SegmentationInfo(BaseModel):
    status: Literal["unstarted", "processing", "ready", "failed"] = "unstarted"
    mask_asset_id: Optional[str] = None
    mask_uri: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    model: Optional[str] = None
    segmented_at: Optional[str] = None
    pixel_count: Optional[int] = None
    error: Optional[str] = None


class SceneEntity(BaseModel):
    id: str = Field(..., min_length=1, description="Unique entity identifier")
    entity_type: EntityType = Field(default="person")
    source_image_id: str = Field(..., min_length=1)
    name: Optional[str] = None
    bounding_box: Optional[BoundingBox2D] = None
    transform: Transform = Field(default_factory=Transform)
    geometry_ref: Optional[str] = None
    material_ref: Optional[str] = None
    skeleton_ref: Optional[str] = None
    animation_state: AnimationState = Field(default_factory=AnimationState)
    visibility: bool = True
    reconstruction_status: ReconstructionStatus = "unstarted"
    confidence: Optional[ConfidenceMetadata] = None
    detection: Optional[DetectionInfo] = None
    segmentation: Optional[SegmentationInfo] = None


class RepresentationAsset(BaseModel):
    status: RepresentationStatus = "unavailable"
    asset_id: Optional[str] = None
    uri: Optional[str] = None
    error: Optional[str] = None
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class SceneRepresentations(BaseModel):
    photo: RepresentationAsset = Field(
        default_factory=lambda: RepresentationAsset(status="ready")
    )
    spatial_photo: RepresentationAsset = Field(
        default_factory=lambda: RepresentationAsset(status="unavailable")
    )
    wireframe: RepresentationAsset = Field(
        default_factory=lambda: RepresentationAsset(status="unavailable")
    )
    solid3d: RepresentationAsset = Field(
        default_factory=lambda: RepresentationAsset(status="unavailable")
    )


class SourceImageMetadata(BaseModel):
    asset_id: str = Field(..., min_length=1)
    filename: str = Field(..., min_length=1)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    aspect_ratio: float = Field(..., gt=0)
    mime_type: str = Field(..., min_length=1)
    size_bytes: int = Field(..., gt=0)
    checksum_sha256: Optional[str] = None
    uploaded_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class Scene(BaseModel):
    id: str = Field(..., min_length=1, description="Unique scene ID")
    schema_version: str = Field(default=SCHEMA_VERSION)
    name: str = Field(..., min_length=1)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source_image: SourceImageMetadata
    coordinate_system: CoordinateSystemInfo = Field(default_factory=CoordinateSystemInfo)
    entities: List[SceneEntity] = Field(default_factory=list)
    representations: SceneRepresentations = Field(default_factory=SceneRepresentations)
    processing_status: SceneProcessingStatus = "unprocessed"
    active_job_id: Optional[str] = None

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema version: '{v}'. Expected '{SCHEMA_VERSION}'")
        return v
