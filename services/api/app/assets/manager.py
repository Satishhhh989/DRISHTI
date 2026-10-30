"""PHOTO → 3D ENGINE - Asset Manager High-Level API
"""

from pathlib import Path
from typing import List, Optional
from app.assets.storage import LocalAssetStore, asset_store
from app.schemas.assets import AssetCategory, AssetMetadata


class AssetManager:
    """Provides high-level registration and resolution of local pipeline assets."""

    def __init__(self, store: Optional[LocalAssetStore] = None):
        self.store = store or asset_store

    def register_photo(
        self,
        content: bytes,
        filename: str,
        mime_type: str = "image/jpeg",
        scene_id: Optional[str] = None
    ) -> AssetMetadata:
        return self.store.store_file(
            content=content,
            category="original_photo",
            original_filename=filename,
            mime_type=mime_type,
            scene_id=scene_id
        )

    def register_pipeline_asset(
        self,
        content: bytes,
        category: AssetCategory,
        filename: str,
        mime_type: str,
        scene_id: str,
        entity_id: Optional[str] = None
    ) -> AssetMetadata:
        """Called by future AI inference stages (segmentation, depth, mesh, etc.)"""
        return self.store.store_file(
            content=content,
            category=category,
            original_filename=filename,
            mime_type=mime_type,
            scene_id=scene_id,
            entity_id=entity_id
        )

    def get_asset(self, asset_id: str) -> AssetMetadata:
        return self.store.get_metadata(asset_id)

    def get_asset_path(self, asset_id: str) -> Path:
        return self.store.get_file_path(asset_id)

    def list_scene_assets(self, scene_id: str) -> List[AssetMetadata]:
        return self.store.list_assets(scene_id=scene_id)


asset_manager = AssetManager()

