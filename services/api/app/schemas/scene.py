"""DRISHTI - Scene Schema Models
Version 1.1.0 (Phase 02: Image Intelligence & Person Segmentation)
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
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


class PlacementProvenance(BaseModel):
    method: str = "depth_projected_composition"
    image_space_coords: List[float] = Field(default_factory=list, description="[normalized_x, normalized_y]")
    depth_source: str = "relative_median_proximity"
    relative_proximity: Optional[float] = None
    model_normalization_applied: str = "canonical_humanoid_scale"
    orientation_correction: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], description="Euler [x, y, z] in radians")
    uncertainty_flags: List[str] = Field(default_factory=list)
    is_metric: bool = False
    placed_at: Optional[str] = None


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


class CoordinateConvention(BaseModel):
    up: str = "+Y"
    forward: str = "+Z"
    right: str = "+X"
    units: str = "meters"


class SkeletonJoint(BaseModel):
    id: str
    name: str
    index: int
    parent_id: Optional[str] = None
    position: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], description="Local position [x, y, z] in meters")
    rotation: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0], description="Local rotation quaternion [qx, qy, qz, qw]")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    is_observed: bool = True


class BoneDefinition(BaseModel):
    id: str
    head_joint_id: str
    tail_joint_id: str
    length: float = 0.0


class SkeletonData(BaseModel):
    id: str
    entity_id: str
    schema_version: str = "1.0.0"
    root_joint_id: str = "pelvis"
    joints: List[SkeletonJoint] = Field(default_factory=list)
    bones: List[BoneDefinition] = Field(default_factory=list)
    coordinate_convention: CoordinateConvention = Field(default_factory=CoordinateConvention)
    source: Literal["smpl_regressed", "yolo_pose_projected", "mock_anatomical"] = "smpl_regressed"
    is_aligned: bool = True
    alignment_transform: Optional[Transform] = None
    validation_status: Literal["valid", "incomplete", "invalid"] = "valid"
    validation_errors: List[str] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class RigData(BaseModel):
    id: str
    entity_id: str
    skeleton_id: str
    mesh_asset_id: Optional[str] = None
    status: Literal["rigged", "unrigged", "failed"] = "unrigged"
    skinning_method: Literal["smpl_lbs_native", "proximity_heat_weights", "none"] = "none"
    bone_count: int = 24
    max_influences_per_vertex: int = 4
    weights_normalized: bool = True
    error_message: Optional[str] = None
    rigged_at: Optional[str] = None


class JointTransformTrack(BaseModel):
    joint_id: str
    timestamps: List[float] = Field(default_factory=list, description="Keyframe time offsets in seconds")
    rotations: List[List[float]] = Field(default_factory=list, description="List of quaternion [qx, qy, qz, qw] per keyframe")
    positions: Optional[List[List[float]]] = None


class AnimationClipData(BaseModel):
    id: str
    name: str
    target_entity_id: str
    target_skeleton_id: str
    duration_seconds: float
    is_looping: bool = True
    fps: float = 30.0
    provenance: Literal["procedural"] = "procedural"
    tracks: List[JointTransformTrack] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


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


class MeshBounds(BaseModel):
    min: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    max: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])


class GeometryMetadata(BaseModel):
    vertex_count: int = Field(default=0, ge=0)
    face_count: int = Field(default=0, ge=0)
    bounds: MeshBounds = Field(default_factory=MeshBounds)
    dimensions: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])


class ModelProvenance(BaseModel):
    name: str = "SPIN"
    version: str = "1.0.0"
    device: str = "cpu"
    checkpoint: Optional[str] = None


class HumanReconstructionData(BaseModel):
    status: Literal["none", "pending", "processing", "ready", "failed"] = "none"
    mesh_asset_id: Optional[str] = None
    mesh_url: Optional[str] = None
    metadata_asset_id: Optional[str] = None
    model_provenance: Optional[ModelProvenance] = None
    geometry_metadata: Optional[GeometryMetadata] = None
    coordinate_convention: Optional[CoordinateConvention] = Field(default_factory=CoordinateConvention)
    transform: Optional[Transform] = None
    texture_availability: Literal["none", "partial", "full"] = "none"
    quality_score: Optional[float] = None
    quality_preset: Optional[str] = "high"
    metric_status: Optional[str] = None
    reprojection_error_pixels: Optional[float] = None
    reprojection_error_normalized: Optional[float] = None
    silhouette_iou: Optional[float] = None
    stance_description: Optional[str] = None
    observed_keypoints_count: Optional[int] = None
    pose_convergence_status: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    reconstructed_at: Optional[str] = None


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
    reconstruction: Optional[HumanReconstructionData] = None
    placement: Optional[PlacementProvenance] = None
    scene_node_id: Optional[str] = None
    skeleton: Optional[SkeletonData] = None
    rig: Optional[RigData] = None
    animation_clips: List[AnimationClipData] = Field(default_factory=list)
    show_skeleton: bool = False


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
    articulated: RepresentationAsset = Field(
        default_factory=lambda: RepresentationAsset(status="unavailable")
    )
    combined: RepresentationAsset = Field(
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


class SceneGraphNode(BaseModel):
    id: str
    name: str
    node_type: Literal["root", "camera", "light", "reference_plane", "ground_plane", "person"]
    entity_id: Optional[str] = None
    transform: Transform = Field(default_factory=Transform)
    children: List[str] = Field(default_factory=list)
    visible: bool = True
    mesh_asset_id: Optional[str] = None
    mesh_uri: Optional[str] = None
    provenance: Optional[PlacementProvenance] = None


class SceneGraph(BaseModel):
    root_id: str = "scene-root"
    nodes: Dict[str, SceneGraphNode] = Field(default_factory=dict)


class SceneAssemblyReport(BaseModel):
    status: Literal["completed", "partial", "failed"] = "completed"
    assembled_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    total_entities: int = 0
    reconstructed_count: int = 0
    unreconstructed_count: int = 0
    occlusion_warnings: List[str] = Field(default_factory=list)
    placement_method: str = "depth_projected_composition"


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
    scene_graph: Optional[SceneGraph] = None
    assembly_report: Optional[SceneAssemblyReport] = None
    processing_status: SceneProcessingStatus = "unprocessed"
    active_job_id: Optional[str] = None

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema version: '{v}'. Expected '{SCHEMA_VERSION}'")
        return v
