"""PHOTO → 3D ENGINE - Asset Metadata Schemas
"""

from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field

AssetCategory = Literal[
    "original_photo",
    "segmentation_mask",
    "depth_map",
    "mesh",
    "texture",
    "skeleton",
    "export",
]


class AssetDimensions(BaseModel):
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)


class AssetMetadata(BaseModel):
    asset_id: str = Field(..., min_length=1)
    category: AssetCategory
    filename: str = Field(..., min_length=1)
    relative_path: str = Field(..., min_length=1)
    mime_type: str = Field(..., min_length=1)
    size_bytes: int = Field(..., gt=0)
    checksum_sha256: str = Field(..., min_length=64, max_length=64)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    scene_id: Optional[str] = None
    entity_id: Optional[str] = None
    dimensions: Optional[AssetDimensions] = None
    format_details: Optional[Dict[str, Any]] = None

