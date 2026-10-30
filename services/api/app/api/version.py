"""PHOTO → 3D ENGINE - Version Endpoint
"""

from fastapi import APIRouter
from pydantic import BaseModel
from app.core.config import settings

router = APIRouter(tags=["System"])


class VersionResponse(BaseModel):
    service_version: str
    api_version: str
    schema_version: str


@router.get("/version", response_model=VersionResponse)
async def get_version() -> VersionResponse:
    return VersionResponse(
        service_version=settings.service_version,
        api_version=settings.api_version,
        schema_version=settings.schema_version,
    )
