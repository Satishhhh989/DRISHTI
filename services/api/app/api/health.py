"""PHOTO → 3D ENGINE - Health Endpoint
"""

from datetime import datetime, timezone
from fastapi import APIRouter
from pydantic import BaseModel
from app.core.config import settings

router = APIRouter(tags=["System"])


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: str
    storage_available: bool


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    storage_ok = settings.storage_root.exists() and settings.storage_root.is_dir()
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        version=settings.service_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        storage_available=storage_ok,
    )

