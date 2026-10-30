"""DRISHTI - Project Schema Models
Version 1.0.0 (Phase 09: Interaction, Persistence & Export)
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
from app.schemas.scene import SCHEMA_VERSION, SourceImageMetadata

ProjectAssetType = Literal["source_photo", "depth_map", "mask", "mesh", "skeleton", "other"]


class ProjectAssetEntry(BaseModel):
    asset_id: str = Field(..., min_length=1)
    asset_type: ProjectAssetType
    relative_path: str = Field(..., min_length=1)
    filename: str = Field(..., min_length=1)
    checksum_sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    mime_type: Optional[str] = None


class ProjectSettings(BaseModel):
    active_representation: str = "solid3d"
    show_grid: bool = True
    show_reference_plane: bool = True


class ProjectProcessingState(BaseModel):
    detection: bool = False
    segmentation: bool = False
    depth: bool = False
    reconstruction: bool = False
    assembly: bool = False
    motion: bool = False


class Project(BaseModel):
    id: str = Field(..., min_length=1, description="Stable unique project ID")
    name: str = Field(..., min_length=1)
    schema_version: str = Field(default=SCHEMA_VERSION)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    scene_id: str = Field(..., min_length=1, description="Associated canonical scene ID")
    source_image: SourceImageMetadata
    settings: ProjectSettings = Field(default_factory=ProjectSettings)
    asset_manifest: List[ProjectAssetEntry] = Field(default_factory=list)
    processing_state: ProjectProcessingState = Field(default_factory=ProjectProcessingState)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema version: '{v}'. Expected '{SCHEMA_VERSION}'")
        return v
