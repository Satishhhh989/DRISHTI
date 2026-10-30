"""PHOTO → 3D ENGINE - Scene Manager

Manages the lifecycle, persistence, and state transitions of 3D scenes.
Validates all incoming mutations against the strict Scene schema.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional
from app.core.errors import NotFoundError, ValidationAppError
from app.core.logging import logger
from app.schemas.assets import AssetMetadata
from app.schemas.scene import (
    CoordinateSystemInfo,
    RepresentationAsset,
    Scene,
    SceneEntity,
    SceneRepresentations,
    SourceImageMetadata,
)


class SceneManager:
    def __init__(self):
        self._scenes: Dict[str, Scene] = {}

    def create_scene_from_photo(
        self,
        photo_asset: AssetMetadata,
        name: Optional[str] = None
    ) -> Scene:
        """Initializes a new Scene anchored on an uploaded source photo."""
        if not photo_asset.dimensions:
            raise ValidationAppError("Source image dimensions could not be determined")

        scene_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        width = photo_asset.dimensions.width
        height = photo_asset.dimensions.height
        aspect = round(width / height, 6)

        source_meta = SourceImageMetadata(
            asset_id=photo_asset.asset_id,
            filename=photo_asset.filename,
            width=width,
            height=height,
            aspect_ratio=aspect,
            mime_type=photo_asset.mime_type,
            size_bytes=photo_asset.size_bytes,
            checksum_sha256=photo_asset.checksum_sha256,
            uploaded_at=now,
        )

        # Initial representations: photo is ready, 3D representations are unavailable
        representations = SceneRepresentations(
            photo=RepresentationAsset(
                status="ready",
                asset_id=photo_asset.asset_id,
                uri=f"/api/v1/assets/{photo_asset.asset_id}/file",
                updated_at=now,
            ),
            spatial_photo=RepresentationAsset(
                status="unavailable",
                updated_at=now,
            ),
            wireframe=RepresentationAsset(
                status="unavailable",
                updated_at=now,
            ),
            solid3d=RepresentationAsset(
                status="unavailable",
                updated_at=now,
            ),
        )

        scene = Scene(
            id=scene_id,
            name=name or f"Scene {photo_asset.filename}",
            created_at=now,
            updated_at=now,
            source_image=source_meta,
            coordinate_system=CoordinateSystemInfo(),
            entities=[],
            representations=representations,
            processing_status="unprocessed",
            active_job_id=None,
        )

        self._scenes[scene_id] = scene
        logger.info(f"Created scene: id={scene_id} name='{scene.name}'")
        return scene

    def get_scene(self, scene_id: str) -> Scene:
        scene = self._scenes.get(scene_id)
        if not scene:
            raise NotFoundError("Scene", scene_id)
        return scene

    def list_scenes(self) -> List[Scene]:
        return list(self._scenes.values())

    def update_scene(self, scene: Scene) -> Scene:
        if scene.id not in self._scenes:
            raise NotFoundError("Scene", scene.id)
        scene.updated_at = datetime.now(timezone.utc).isoformat()
        self._scenes[scene.id] = scene
        return scene


scene_manager = SceneManager()
