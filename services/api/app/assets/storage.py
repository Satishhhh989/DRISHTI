"""PHOTO → 3D ENGINE - Local Asset Storage Engine

Guarantees path safety, prevents directory traversal, and enforces
category-based file separation under the configured local storage root.
"""

import hashlib
import io
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image

from app.core.config import settings
from app.core.errors import NotFoundError, StorageSecurityError
from app.core.logging import logger
from app.schemas.assets import AssetCategory, AssetDimensions, AssetMetadata

CATEGORY_DIRS: Dict[AssetCategory, str] = {
    "original_photo": "photos",
    "segmentation_mask": "masks",
    "depth_map": "depth",
    "mesh": "meshes",
    "texture": "textures",
    "skeleton": "skeletons",
    "export": "exports",
}


def sanitize_filename(name: str) -> str:
    """Removes path separators and non-whitelisted characters."""
    clean = Path(name).name
    clean = re.sub(r"[^\w\.\-]", "_", clean)
    return clean or "asset.bin"


class LocalAssetStore:
    """Local-first filesystem storage manager with path traversal protection."""

    def __init__(self, root: Optional[Path] = None):
        self.root = (root or settings.storage_root).resolve()
        self._registry: Dict[str, AssetMetadata] = {}
        self._init_directories()

    def _init_directories(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in CATEGORY_DIRS.values():
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(self, relative_path: Path) -> Path:
        """Resolves target and strictly verifies that it resides within storage root."""
        target = (self.root / relative_path).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as err:
            logger.warning(f"Path traversal attempt prevented: {relative_path}")
            raise StorageSecurityError("Access outside the designated storage root is strictly prohibited") from err
        return target

    def store_file(
        self,
        content: bytes,
        category: AssetCategory,
        original_filename: str,
        mime_type: str,
        scene_id: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> AssetMetadata:
        """Stores a binary asset file safely and records its validated metadata."""
        if len(content) > settings.max_upload_size_bytes:
            raise ValueError(
                f"File size exceeds maximum permitted limit ({settings.max_upload_size_mb} MB)"
            )

        asset_id = str(uuid.uuid4())
        clean_name = sanitize_filename(original_filename)
        dir_name = CATEGORY_DIRS.get(category, "exports")
        relative_path = Path(dir_name) / f"{asset_id}_{clean_name}"
        target_path = self._resolve_safe_path(relative_path)

        # Compute SHA256 checksum
        checksum = hashlib.sha256(content).hexdigest()

        # Write file
        target_path.write_bytes(content)

        # Inspect dimensions for images
        dimensions: Optional[AssetDimensions] = None
        if category in ("original_photo", "segmentation_mask", "depth_map", "texture"):
            try:
                with Image.open(io.BytesIO(content)) as img:
                    dimensions = AssetDimensions(width=img.width, height=img.height)
            except Exception as e:
                logger.debug(f"Could not extract image dimensions for {original_filename}: {e}")

        meta = AssetMetadata(
            asset_id=asset_id,
            category=category,
            filename=original_filename,
            relative_path=str(relative_path),
            mime_type=mime_type,
            size_bytes=len(content),
            checksum_sha256=checksum,
            scene_id=scene_id,
            entity_id=entity_id,
            dimensions=dimensions,
        )

        self._registry[asset_id] = meta
        logger.info(f"Stored local asset: id={asset_id} category={category} size={len(content)}B")
        return meta

    def get_metadata(self, asset_id: str) -> AssetMetadata:
        meta = self._registry.get(asset_id)
        if not meta:
            raise NotFoundError("Asset", asset_id)
        return meta

    def get_file_path(self, asset_id: str) -> Path:
        meta = self.get_metadata(asset_id)
        target = self._resolve_safe_path(Path(meta.relative_path))
        if not target.exists():
            raise NotFoundError("Asset file", asset_id)
        return target

    def list_assets(
        self,
        scene_id: Optional[str] = None,
        category: Optional[AssetCategory] = None
    ) -> List[AssetMetadata]:
        results = list(self._registry.values())
        if scene_id:
            results = [a for a in results if a.scene_id == scene_id]
        if category:
            results = [a for a in results if a.category == category]
        return results


asset_store = LocalAssetStore()
