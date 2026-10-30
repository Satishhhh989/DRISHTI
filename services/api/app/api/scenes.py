"""PHOTO → 3D ENGINE - Scene Endpoints
"""

from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel

from app.assets.manager import asset_manager
from app.scene.manager import scene_manager
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

