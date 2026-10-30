"""PHOTO → 3D ENGINE - Scene Manager

Manages the lifecycle, persistence, and state transitions of 3D scenes.
Validates all incoming mutations against the strict Scene schema.
Persists assembled scenes locally to data/storage/scenes/{scene_id}.json.
"""

import json
from pathlib import Path
import re
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional
from app.core.config import settings
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

SAFE_SCENE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


class SceneManager:
    def __init__(self, storage_dir: Optional[Path] = None):
        self._scenes: Dict[str, Scene] = {}
        self.storage_dir = storage_dir or (settings.storage_root / "scenes")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_scene_path(self, scene_id: str) -> Path:
        if not SAFE_SCENE_ID_PATTERN.match(scene_id):
            raise ValidationAppError(f"Invalid scene identifier: '{scene_id}'")
        return self.storage_dir / f"{scene_id}.json"

    def _persist_to_disk(self, scene: Scene) -> None:
        try:
            path = self._get_scene_path(scene.id)
            tmp_path = path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(scene.model_dump_json(indent=2))
            tmp_path.replace(path)
            logger.debug(f"Persisted scene {scene.id} to {path}")
        except Exception as e:
            logger.warning(f"Failed to persist scene {scene.id} to disk: {e}")

    def _load_from_disk(self, scene_id: str) -> Optional[Scene]:
        try:
            path = self._get_scene_path(scene_id)
            if not path.is_file():
                return None
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            scene = Scene.model_validate(data)
            self._scenes[scene.id] = scene
            return scene
        except Exception as e:
            logger.warning(f"Failed to load scene {scene_id} from disk: {e}")
            return None

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
        self._persist_to_disk(scene)
        logger.info(f"Created scene: id={scene_id} name='{scene.name}'")
        return scene

    def get_scene(self, scene_id: str) -> Scene:
        scene = self._scenes.get(scene_id)
        if not scene:
            scene = self._load_from_disk(scene_id)
        if not scene:
            raise NotFoundError("Scene", scene_id)
        return scene

    def list_scenes(self) -> List[Scene]:
        # Also discover any scenes persisted on disk that aren't yet cached in memory
        try:
            for p in self.storage_dir.glob("*.json"):
                sid = p.stem
                if sid not in self._scenes:
                    self._load_from_disk(sid)
        except Exception as e:
            logger.warning(f"Error listing scenes from disk: {e}")
        return list(self._scenes.values())

    def update_scene(self, scene: Scene) -> Scene:
        if scene.id not in self._scenes and not self._get_scene_path(scene.id).is_file():
            raise NotFoundError("Scene", scene.id)
        scene.updated_at = datetime.now(timezone.utc).isoformat()
        self._scenes[scene.id] = scene
        self._persist_to_disk(scene)
        return scene


scene_manager = SceneManager()

