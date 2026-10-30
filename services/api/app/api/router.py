"""PHOTO → 3D ENGINE - API v1 Router
"""

from fastapi import APIRouter
from app.api import assets, events, health, scenes, version

api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(health.router)
api_v1_router.include_router(version.router)
api_v1_router.include_router(assets.router)
api_v1_router.include_router(scenes.router)
api_v1_router.include_router(events.router)
