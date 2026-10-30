"""PHOTO → 3D ENGINE - Local Asset Endpoints
"""

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from app.assets.manager import asset_manager
from app.assets.storage import sanitize_filename
from app.core.errors import NotFoundError
from app.scene.manager import scene_manager
from app.schemas.assets import AssetMetadata
from app.schemas.scene import Scene
from pathlib import Path

router = APIRouter(prefix="/assets", tags=["Assets"])


class UploadAssetResponse(BaseModel):
    asset: AssetMetadata
    scene: Optional[Scene] = None


@router.post("/upload", response_model=UploadAssetResponse)
async def upload_photo(
    file: UploadFile = File(...),
    auto_create_scene: bool = Form(default=True),
    scene_name: Optional[str] = Form(default=None)
) -> UploadAssetResponse:
    content = await file.read()
    mime = file.content_type or "image/jpeg"
    filename = file.filename or "uploaded_photo.jpg"

    asset = asset_manager.register_photo(
        content=content,
        filename=filename,
        mime_type=mime
    )

    created_scene: Optional[Scene] = None
    if auto_create_scene and asset.dimensions:
        created_scene = scene_manager.create_scene_from_photo(
            photo_asset=asset,
            name=scene_name
        )

    return UploadAssetResponse(asset=asset, scene=created_scene)


@router.get("/{asset_id}", response_model=AssetMetadata)
async def get_asset_metadata(asset_id: str) -> AssetMetadata:
    return asset_manager.get_asset(asset_id)


@router.get("/{asset_id}/file")
async def get_asset_file(asset_id: str):
    file_path = asset_manager.get_asset_path(asset_id)
    meta = asset_manager.get_asset(asset_id)
    return FileResponse(
        path=str(file_path),
        media_type=meta.mime_type,
        filename=meta.filename
    )


@router.get("/meshes/{filename}")
async def get_mesh_file(filename: str):
    safe_name = sanitize_filename(filename)
    path = asset_manager.store._resolve_safe_path(Path("meshes") / safe_name)
    if not path.exists():
        raise NotFoundError(f"Mesh '{filename}' not found.")
    return FileResponse(
        path=str(path),
        media_type="model/gltf-binary",
        filename=safe_name
    )

