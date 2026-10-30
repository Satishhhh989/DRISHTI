"""PHOTO → 3D ENGINE - Scene Endpoints
"""

from typing import List, Optional
import math
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.assets.manager import asset_manager
from app.core.errors import NotFoundError, ValidationAppError
from app.scene.manager import scene_manager
from app.schemas.common import Transform, Vector3D, EulerRotation
from app.schemas.scene import Scene

router = APIRouter(prefix="/scenes", tags=["Scenes"])


class CreateSceneRequest(BaseModel):
    source_asset_id: str
    name: Optional[str] = None


class SceneListResponse(BaseModel):
    scenes: List[Scene]
    total: int


@router.get("", response_model=SceneListResponse)
async def list_scenes() -> SceneListResponse:
    scenes = scene_manager.list_scenes()
    return SceneListResponse(scenes=scenes, total=len(scenes))


@router.post("", response_model=Scene)
async def create_scene(request: CreateSceneRequest) -> Scene:
    asset = asset_manager.get_asset(request.source_asset_id)
    return scene_manager.create_scene_from_photo(
        photo_asset=asset,
        name=request.name
    )


@router.get("/{scene_id}", response_model=Scene)
async def get_scene(scene_id: str) -> Scene:
    return scene_manager.get_scene(scene_id)


@router.put("/{scene_id}", response_model=Scene)
async def update_full_scene(scene_id: str, updated_scene: Scene) -> Scene:
    if updated_scene.id != scene_id:
        raise ValidationAppError(f"Scene ID in path '{scene_id}' does not match body '{updated_scene.id}'")
    return scene_manager.update_scene(updated_scene)


class UpdateTransformRequest(BaseModel):
    position: Optional[Vector3D] = None
    rotation: Optional[EulerRotation] = None
    scale: Optional[Vector3D] = None


def validate_transform_bounds(t: Transform) -> None:
    # Position
    for axis, val in [("x", t.position.x), ("y", t.position.y), ("z", t.position.z)]:
        if not math.isfinite(val):
            raise ValidationAppError(f"Position.{axis} must be finite, received {val}")
        if abs(val) > 50.0:
            raise ValidationAppError(f"Position.{axis} exceeds permissible scene bounds [-50, 50] meters: {val}")

    # Rotation
    for axis, val in [("x", t.rotation.x), ("y", t.rotation.y), ("z", t.rotation.z)]:
        if not math.isfinite(val):
            raise ValidationAppError(f"Rotation.{axis} must be finite, received {val}")

    # Scale
    for axis, val in [("x", t.scale.x), ("y", t.scale.y), ("z", t.scale.z)]:
        if not math.isfinite(val) or val <= 0:
            raise ValidationAppError(f"Scale.{axis} must be positive finite, received {val}")
        if val < 0.05 or val > 10.0:
            raise ValidationAppError(f"Scale.{axis} out of permissible range [0.05, 10.0]: {val}")


@router.patch("/{scene_id}/entities/{entity_id}/transform", response_model=Scene)
async def update_entity_transform(
    scene_id: str,
    entity_id: str,
    request: UpdateTransformRequest
) -> Scene:
    scene = scene_manager.get_scene(scene_id)
    target_entity = None
    for entity in scene.entities:
        if entity.id == entity_id:
            target_entity = entity
            break

    if not target_entity:
        raise NotFoundError(f"Entity '{entity_id}' not found in scene '{scene_id}'")

    current = target_entity.transform
    new_pos = request.position or current.position
    new_rot = request.rotation or current.rotation
    new_scale = request.scale or current.scale

    try:
        new_transform = Transform(position=new_pos, rotation=new_rot, scale=new_scale)
        validate_transform_bounds(new_transform)
    except Exception as e:
        raise ValidationAppError(f"Invalid transform specification: {e}")

    target_entity.transform = new_transform
    return scene_manager.update_scene(scene)


@router.post("/{scene_id}/entities/{entity_id}/reset-transform", response_model=Scene)
async def reset_entity_transform(scene_id: str, entity_id: str) -> Scene:
    scene = scene_manager.get_scene(scene_id)
    target_entity = None
    for entity in scene.entities:
        if entity.id == entity_id:
            target_entity = entity
            break

    if not target_entity:
        raise NotFoundError(f"Entity '{entity_id}' not found in scene '{scene_id}'")

    # Reset to initial assembled transform
    if target_entity.placement:
        # Initial assembled transform
        img_coords = target_entity.placement.image_space_coords or [0.5, 0.5]
        # Reconstruct standard placement position
        x = (img_coords[0] - 0.5) * 2.0
        z = -1.5 if target_entity.placement.relative_proximity is None else -0.5 - (1.0 - target_entity.placement.relative_proximity) * 3.0
        y = 0.0
        rot = target_entity.placement.orientation_correction or [0.0, 0.0, 0.0]
        target_entity.transform = Transform(
            position=Vector3D(x=round(x, 4), y=y, z=round(z, 4)),
            rotation=EulerRotation(x=rot[0], y=rot[1], z=rot[2]),
            scale=Vector3D(x=1.0, y=1.0, z=1.0)
        )
    else:
        target_entity.transform = Transform(
            position=Vector3D(x=0.0, y=0.0, z=-1.5),
            rotation=EulerRotation(x=0.0, y=0.0, z=0.0),
            scale=Vector3D(x=1.0, y=1.0, z=1.0)
        )

    return scene_manager.update_scene(scene)

