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


class EntityDepthInfo(BaseModel):
    status: Literal["unstarted", "processing", "ready", "failed"] = "unstarted"
    median_proximity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Canonical relative proximity in [0, 1] where 1.0 is nearest foreground and 0.0 is farthest background",
    )
    min_proximity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_proximity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    depth_variation: Optional[float] = None
    valid_pixel_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    method: Literal["segmentation_mask", "bounding_box"] = "segmentation_mask"
    depth_asset_id: Optional[str] = None
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
    depth: Optional[EntityDepthInfo] = None


class DepthNormalizationMetadata(BaseModel):
    method: str = "percentile_clamped"
    p_min: float
    p_max: float
    raw_min: float
    raw_max: float
    model_convention: str = "distance_smaller_is_closer"
    canonical_convention: str = "relative_proximity"


class SceneDepthMetadata(BaseModel):
    status: Literal["unavailable", "pending", "processing", "ready", "failed"] = "unavailable"
    depth_asset_id: Optional[str] = None
    depth_uri: Optional[str] = None
    visualization_asset_id: Optional[str] = None
    visualization_uri: Optional[str] = None
    model: str = "depth-anything-v2-small"
    model_family: str = "Depth-Anything-V2"
    depth_convention: str = "relative_proximity"
    normalization: Optional[DepthNormalizationMetadata] = None
    width: int = Field(default=0, ge=0)
    height: int = Field(default=0, ge=0)
    device: str = "cpu"
    analyzed_at: Optional[str] = None
    error: Optional[str] = None


class SpatialPhotoConfig(BaseModel):
    default_depth_strength: float = 0.08
    geometry_subdivisions: List[int] = Field(default_factory=lambda: [128, 128])
    disocclusion_handling: str = "conservative_bounds"


class RepresentationAsset(BaseModel):
    status: RepresentationStatus = "unavailable"
    asset_id: Optional[str] = None
    uri: Optional[str] = None
    error: Optional[str] = None
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    config: Optional[SpatialPhotoConfig] = None


class SceneRepresentations(BaseModel):
    photo: RepresentationAsset = Field(
        default_factory=lambda: RepresentationAsset(status="ready")
    )
    spatial_photo: RepresentationAsset = Field(
        default_factory=lambda: RepresentationAsset(
            status="unavailable",
            config=SpatialPhotoConfig(),
        )
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
    depth: Optional[SceneDepthMetadata] = None
    processing_status: SceneProcessingStatus = "unprocessed"
    active_job_id: Optional[str] = None

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema version: '{v}'. Expected '{SCHEMA_VERSION}'")
        return v
