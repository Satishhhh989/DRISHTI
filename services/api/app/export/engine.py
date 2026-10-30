"""DRISHTI - Real Export Engine
Version 1.0.0 (Phase 09: Interaction, Persistence & Export)

Implements genuine local asset export for:
1. Single entity reconstructed mesh (GLB with verified geometry).
2. Assembled multi-person 3D scene (GLB preserving node transforms and hierarchy).
3. Canonical scene metadata (JSON).
4. Portable self-contained project archive (ZIP with manifest and assets).
5. Safe project archive import with strict path traversal (anti-zip-slip) defenses.
"""

import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid
import zipfile

import numpy as np
import trimesh

from app.assets.manager import asset_manager
from app.core.config import settings
from app.core.errors import NotFoundError, ValidationAppError
from app.core.logging import logger
from app.project.manager import project_manager
from app.scene.manager import scene_manager
from app.schemas.project import Project, ProjectAssetEntry
from app.schemas.scene import Scene, SceneEntity


def euler_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Computes standard 4x4 rotation matrix from XYZ Euler angles in radians."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    # R = Rx * Ry * Rz (matching Three.js default XYZ order)
    rot = np.array([
        [cy * cz, -cy * sz, sy, 0.0],
        [cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy, 0.0],
        [sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64)
    return rot


def build_transform_matrix(entity: SceneEntity) -> np.ndarray:
    """Builds 4x4 affine transform matrix (TRS: Translation * Rotation * Scale)."""
    t = entity.transform
    mat = np.eye(4, dtype=np.float64)

    # Scale
    s_mat = np.diag([t.scale.x, t.scale.y, t.scale.z, 1.0])

    # Rotation
    r_mat = euler_to_matrix(t.rotation.x, t.rotation.y, t.rotation.z)

    # Translation
    t_mat = np.eye(4, dtype=np.float64)
    t_mat[0, 3] = t.position.x
    t_mat[1, 3] = t.position.y
    t_mat[2, 3] = t.position.z

    mat = t_mat @ r_mat @ s_mat
    return mat


class ExportEngine:
    def export_entity_mesh_glb(
        self,
        scene_id: str,
        entity_id: str,
        apply_transform: bool = False
    ) -> bytes:
        """Exports a single person's reconstructed mesh to GLB format."""
        scene = scene_manager.get_scene(scene_id)
        target: Optional[SceneEntity] = None
        for e in scene.entities:
            if e.id == entity_id:
                target = e
                break

        if not target:
            raise NotFoundError(f"Entity '{entity_id}' not found in scene '{scene_id}'")

        if (
            not target.reconstruction
            or target.reconstruction.status != "ready"
            or not target.reconstruction.mesh_asset_id
        ):
            raise ValidationAppError(
                f"Entity '{entity_id}' has no reconstructed 3D mesh. "
                "Placeholder geometry cannot be exported as a real reconstruction."
            )

        # Locate mesh file
        asset_id = target.reconstruction.mesh_asset_id
        mesh_path = asset_manager.get_asset_path(asset_id)
        if not mesh_path.exists():
            raise NotFoundError(f"Mesh asset file '{asset_id}' not found on disk")

        # Load and validate with trimesh
        loaded = trimesh.load(str(mesh_path), file_type="glb")
        if isinstance(loaded, trimesh.Scene):
            geoms = list(loaded.geometry.values())
            if not geoms:
                raise ValidationAppError(f"Mesh file '{asset_id}' contains no geometry.")
            mesh = geoms[0]
        else:
            mesh = loaded

        if apply_transform:
            mat = build_transform_matrix(target)
            mesh.apply_transform(mat)

        # Export and validate output GLB
        glb_bytes = trimesh.exchange.gltf.export_glb(mesh)
        test_load = trimesh.load(trimesh.util.wrap_as_stream(glb_bytes), file_type="glb")
        if test_load is None:
            raise ValidationAppError("Generated GLB failed internal validation.")

        return glb_bytes

    def export_assembled_scene_glb(self, scene_id: str) -> bytes:
        """Exports the entire multi-person scene to an assembled GLB with node transforms."""
        scene = scene_manager.get_scene(scene_id)
        valid_entities: List[SceneEntity] = [
            e for e in scene.entities
            if e.reconstruction and e.reconstruction.status == "ready" and e.reconstruction.mesh_asset_id
        ]

        if not valid_entities:
            raise ValidationAppError(
                f"Scene '{scene_id}' has no reconstructed 3D meshes to export."
            )

        export_scene = trimesh.Scene()

        for entity in valid_entities:
            asset_id = entity.reconstruction.mesh_asset_id  # type: ignore
            try:
                mesh_path = asset_manager.get_asset_path(asset_id)
                if not mesh_path.exists():
                    logger.warning(f"Skipping missing mesh asset {asset_id} for entity {entity.id}")
                    continue

                loaded = trimesh.load(str(mesh_path), file_type="glb")
                if isinstance(loaded, trimesh.Scene):
                    geoms = list(loaded.geometry.values())
                    if not geoms:
                        continue
                    m = geoms[0].copy()
                else:
                    m = loaded.copy()

                mat = build_transform_matrix(entity)
                export_scene.add_geometry(
                    m,
                    node_name=f"{entity.name or entity.id}_{entity.id}",
                    transform=mat
                )
            except Exception as e:
                logger.warning(f"Error packing entity {entity.id} into scene export: {e}")

        if len(export_scene.geometry) == 0:
            raise ValidationAppError("No valid geometries could be packed into the assembled GLB.")

        glb_bytes = trimesh.exchange.gltf.export_glb(export_scene)
        test_load = trimesh.load(trimesh.util.wrap_as_stream(glb_bytes), file_type="glb")
        if test_load is None:
            raise ValidationAppError("Generated assembled GLB failed internal validation.")

        return glb_bytes

    def export_scene_metadata_json(self, scene_id: str) -> str:
        """Exports canonical scene metadata formatted as standard JSON."""
        scene = scene_manager.get_scene(scene_id)
        return scene.model_dump_json(indent=2)

    def create_portable_project_archive(self, project_id: str) -> bytes:
        """Creates a portable self-contained ZIP archive of the project and its assets."""
        project = project_manager.get_project(project_id)
        scene = scene_manager.get_scene(project.scene_id)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # 1. Project metadata
            zf.writestr("project.json", project.model_dump_json(indent=2))

            # 2. Scene metadata
            zf.writestr("scene.json", scene.model_dump_json(indent=2))

            # 3. Asset manifest
            manifest_data = [m.model_dump() for m in project.asset_manifest]
            zf.writestr("manifest.json", json.dumps(manifest_data, indent=2))

            # 4. Pack asset files
            for entry in project.asset_manifest:
                try:
                    asset_path = asset_manager.get_asset_path(entry.asset_id)
                    if asset_path.exists() and asset_path.is_file():
                        # Store using relative path
                        zf.write(asset_path, arcname=entry.relative_path)
                except Exception as e:
                    logger.warning(f"Could not pack asset {entry.asset_id} into archive: {e}")

        zip_bytes = zip_buffer.getvalue()
        return zip_bytes

    def import_portable_project_archive(
        self,
        zip_bytes: bytes,
        new_name: Optional[str] = None
    ) -> Project:
        """Imports and validates a portable DRISHTI project archive.
        
        Strictly defends against Path Traversal (Anti-Zip-Slip), validates manifests
        and schemas, and securely stores assets into local storage.
        """
        if not zip_bytes:
            raise ValidationAppError("Archive bytes are empty.")

        try:
            zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        except Exception as e:
            raise ValidationAppError(f"Invalid ZIP archive: {e}")

        # Security check: verify no path traversal in any entry
        for member in zf.infolist():
            filename = member.filename
            if filename.startswith("/") or filename.startswith("\\") or ":" in filename:
                raise ValidationAppError(
                    f"Security violation: Archive contains absolute or illegal path: '{filename}'"
                )
            parts = Path(filename).parts
            if ".." in parts:
                raise ValidationAppError(
                    f"Security violation: Archive contains path traversal components: '{filename}'"
                )

        namelist = set(zf.namelist())
        if "project.json" not in namelist:
            raise ValidationAppError("Archive missing required 'project.json'")
        if "scene.json" not in namelist:
            raise ValidationAppError("Archive missing required 'scene.json'")

        # Parse and validate project and scene JSON
        try:
            proj_data = json.loads(zf.read("project.json").decode("utf-8"))
            scene_data = json.loads(zf.read("scene.json").decode("utf-8"))
        except Exception as e:
            raise ValidationAppError(f"Could not parse JSON files in archive: {e}")

        # Generate fresh safe identifiers to prevent accidental collisions
        new_proj_id = f"proj-imp-{uuid.uuid4().hex[:8]}"
        new_scene_id = f"scene-imp-{uuid.uuid4().hex[:8]}"

        proj_data["id"] = new_proj_id
        if new_name:
            proj_data["name"] = new_name
        else:
            proj_data["name"] = f"{proj_data.get('name', 'Imported Project')} (Imported)"

        scene_data["id"] = new_scene_id
        proj_data["scene_id"] = new_scene_id

        # Validate with Pydantic
        imported_scene = Scene.model_validate(scene_data)
        imported_project = Project.model_validate(proj_data)

        # Extract assets into asset storage
        for entry in imported_project.asset_manifest:
            arc_path = entry.relative_path
            if arc_path in namelist:
                try:
                    content = zf.read(arc_path)
                    asset_manager.store.save_file(
                        category=Path(entry.relative_path).parent.name or "misc",
                        filename=entry.filename,
                        content=content
                    )
                except Exception as e:
                    logger.warning(f"Error extracting asset {arc_path}: {e}")

        # Persist scene and project
        scene_manager._scenes[new_scene_id] = imported_scene
        scene_manager._persist_to_disk(imported_scene)

        project_manager._projects[new_proj_id] = imported_project
        project_manager._persist_to_disk(imported_project)

        logger.info(f"Successfully imported project: id={new_proj_id} name='{imported_project.name}'")
        return imported_project


export_engine = ExportEngine()

