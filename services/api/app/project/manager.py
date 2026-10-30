"""DRISHTI - Local Project Manager
Version 1.0.0 (Phase 09: Interaction, Persistence & Export)

Manages persistent projects, asset manifests, atomic writes, and project lifecycle.
Stores project metadata in data/storage/projects/{project_id}.json.
"""

import json
from pathlib import Path
import re
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from app.assets.manager import asset_manager
from app.core.config import settings
from app.core.errors import NotFoundError, ValidationAppError
from app.core.logging import logger
from app.scene.manager import scene_manager
from app.schemas.project import (
    Project,
    ProjectAssetEntry,
    ProjectProcessingState,
    ProjectSettings,
)
from app.schemas.scene import Scene

SAFE_PROJECT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


class ProjectManager:
    def __init__(self, storage_dir: Optional[Path] = None):
        self._projects: Dict[str, Project] = {}
        self.storage_dir = storage_dir or (settings.storage_root / "projects")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_project_path(self, project_id: str) -> Path:
        if not SAFE_PROJECT_ID_PATTERN.match(project_id):
            raise ValidationAppError(f"Invalid project identifier: '{project_id}'")
        return self.storage_dir / f"{project_id}.json"

    def _persist_to_disk(self, project: Project) -> None:
        """Atomically writes project metadata to disk via temporary file."""
        try:
            path = self._get_project_path(project.id)
            tmp_path = path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(project.model_dump_json(indent=2))
                f.flush()
            tmp_path.replace(path)
            logger.debug(f"Atomically persisted project {project.id} to {path}")
        except Exception as e:
            logger.error(f"Failed to persist project {project.id} to disk: {e}")
            raise ValidationAppError(f"Could not save project to disk: {e}")

    def _load_from_disk(self, project_id: str) -> Optional[Project]:
        try:
            path = self._get_project_path(project_id)
            if not path.is_file():
                return None
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            project = Project.model_validate(data)
            self._projects[project.id] = project
            return project
        except Exception as e:
            logger.warning(f"Failed to load project {project_id} from disk: {e}")
            return None

    def build_asset_manifest(self, scene: Scene) -> List[ProjectAssetEntry]:
        """Resolves all assets referenced across the scene into a structured manifest."""
        manifest: List[ProjectAssetEntry] = []
        seen_assets = set()

        # 1. Source photo
        if scene.source_image and scene.source_image.asset_id:
            aid = scene.source_image.asset_id
            if aid not in seen_assets:
                seen_assets.add(aid)
                try:
                    meta = asset_manager.get_asset(aid)
                    manifest.append(
                        ProjectAssetEntry(
                            asset_id=aid,
                            asset_type="source_photo",
                            relative_path=f"assets/photos/{meta.filename}",
                            filename=meta.filename,
                            checksum_sha256=meta.checksum_sha256,
                            size_bytes=meta.size_bytes,
                            mime_type=meta.mime_type,
                        )
                    )
                except Exception:
                    manifest.append(
                        ProjectAssetEntry(
                            asset_id=aid,
                            asset_type="source_photo",
                            relative_path=f"assets/photos/{scene.source_image.filename}",
                            filename=scene.source_image.filename,
                            checksum_sha256=scene.source_image.checksum_sha256,
                            size_bytes=scene.source_image.size_bytes,
                            mime_type=scene.source_image.mime_type,
                        )
                    )

        # 2. Depth map assets
        if scene.depth:
            if scene.depth.depth_asset_id and scene.depth.depth_asset_id not in seen_assets:
                aid = scene.depth.depth_asset_id
                seen_assets.add(aid)
                try:
                    meta = asset_manager.get_asset(aid)
                    manifest.append(
                        ProjectAssetEntry(
                            asset_id=aid,
                            asset_type="depth_map",
                            relative_path=f"assets/depth/{meta.filename}",
                            filename=meta.filename,
                            checksum_sha256=meta.checksum_sha256,
                            size_bytes=meta.size_bytes,
                            mime_type=meta.mime_type,
                        )
                    )
                except Exception:
                    pass

        # 3. Entity assets (masks, meshes)
        for entity in scene.entities:
            # Segmentation mask
            if entity.segmentation and entity.segmentation.mask_asset_id:
                aid = entity.segmentation.mask_asset_id
                if aid not in seen_assets:
                    seen_assets.add(aid)
                    try:
                        meta = asset_manager.get_asset(aid)
                        manifest.append(
                            ProjectAssetEntry(
                                asset_id=aid,
                                asset_type="mask",
                                relative_path=f"assets/masks/{meta.filename}",
                                filename=meta.filename,
                                checksum_sha256=meta.checksum_sha256,
                                size_bytes=meta.size_bytes,
                                mime_type=meta.mime_type,
                            )
                        )
                    except Exception:
                        pass

            # 3D Reconstructed Mesh
            if entity.reconstruction and entity.reconstruction.mesh_asset_id:
                aid = entity.reconstruction.mesh_asset_id
                if aid not in seen_assets:
                    seen_assets.add(aid)
                    try:
                        meta = asset_manager.get_asset(aid)
                        manifest.append(
                            ProjectAssetEntry(
                                asset_id=aid,
                                asset_type="mesh",
                                relative_path=f"assets/meshes/{meta.filename}",
                                filename=meta.filename,
                                checksum_sha256=meta.checksum_sha256,
                                size_bytes=meta.size_bytes,
                                mime_type=meta.mime_type,
                            )
                        )
                    except Exception:
                        pass

        return manifest

    def compute_processing_state(self, scene: Scene) -> ProjectProcessingState:
        return ProjectProcessingState(
            detection=any(e.detection is not None for e in scene.entities),
            segmentation=any(
                e.segmentation is not None and e.segmentation.status == "ready"
                for e in scene.entities
            ),
            depth=scene.depth is not None and scene.depth.status == "ready",
            reconstruction=any(
                e.reconstruction is not None and e.reconstruction.status == "ready"
                for e in scene.entities
            ),
            assembly=scene.assembly_report is not None and scene.assembly_report.status in ("completed", "partial"),
            motion=any(len(e.animation_clips) > 0 for e in scene.entities),
        )

    def create_project_from_scene(
        self,
        scene: Scene,
        name: Optional[str] = None,
        settings: Optional[ProjectSettings] = None,
    ) -> Project:
        """Initializes and persists a new Project referencing an active Scene."""
        project_id = f"proj-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        manifest = self.build_asset_manifest(scene)
        state = self.compute_processing_state(scene)

        project = Project(
            id=project_id,
            name=name or f"Project {scene.name}",
            created_at=now,
            updated_at=now,
            scene_id=scene.id,
            source_image=scene.source_image,
            settings=settings or ProjectSettings(),
            asset_manifest=manifest,
            processing_state=state,
        )

        self._projects[project_id] = project
        self._persist_to_disk(project)
        logger.info(f"Created project: id={project_id} name='{project.name}'")
        return project

    def get_project(self, project_id: str) -> Project:
        project = self._projects.get(project_id)
        if not project:
            project = self._load_from_disk(project_id)
        if not project:
            raise NotFoundError("Project", project_id)
        return project

    def list_projects(self) -> List[Project]:
        try:
            for p in self.storage_dir.glob("*.json"):
                pid = p.stem
                if pid not in self._projects:
                    self._load_from_disk(pid)
        except Exception as e:
            logger.warning(f"Error scanning projects from disk: {e}")

        # Return sorted by updated_at descending
        return sorted(
            self._projects.values(),
            key=lambda prj: prj.updated_at,
            reverse=True,
        )

    def save_project(
        self,
        project_id: str,
        name: Optional[str] = None,
        settings: Optional[ProjectSettings] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Project:
        """Refreshes scene manifest, updates timestamps, and atomically persists project."""
        project = self.get_project(project_id)
        now = datetime.now(timezone.utc).isoformat()

        # Refresh scene references and manifest
        try:
            scene = scene_manager.get_scene(project.scene_id)
            project.asset_manifest = self.build_asset_manifest(scene)
            project.processing_state = self.compute_processing_state(scene)
        except Exception as e:
            logger.warning(f"Could not refresh scene for project {project_id}: {e}")

        if name:
            project.name = name
        if settings:
            project.settings = settings
        if metadata:
            project.metadata.update(metadata)

        project.updated_at = now
        self._projects[project_id] = project
        self._persist_to_disk(project)
        return project

    def delete_project(self, project_id: str) -> bool:
        project = self.get_project(project_id)
        path = self._get_project_path(project.id)
        if path.is_file():
            path.unlink()
        self._projects.pop(project_id, None)
        return True


project_manager = ProjectManager()

