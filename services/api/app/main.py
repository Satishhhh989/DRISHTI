"""PHOTO → 3D ENGINE - FastAPI Application Entrypoint

Local-first backend service. Runs on loopback (127.0.0.1) by default.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_v1_router
from app.core.config import settings
from app.core.errors import (
    AppException,
    app_exception_handler,
    generic_exception_handler,
)
from app.core.logging import logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_logging(settings.api_log_level)
    logger.info(
        f"Starting {settings.service_name} v{settings.service_version} on {settings.api_host}:{settings.api_port}"
    )
    # Ensure storage root exists
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Local storage root validated: {settings.storage_root.resolve()}")
    yield
    logger.info(f"Shutting down {settings.service_name}")


def create_app() -> FastAPI:
    app = FastAPI(
        title="PHOTO → 3D ENGINE API",
        description="Local-first spatial reconstruction backend foundation",
        version=settings.service_version,
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception Handlers
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # Security Headers Middleware
    @app.middleware("http")
    async def add_security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response

    # Mount API v1 router
    app.include_router(api_v1_router)

    @app.get("/", tags=["System"])
    async def root():
        return {
            "name": settings.service_name,
            "version": settings.service_version,
            "status": "online",
            "docs_url": "/docs",
        }

    return app


app = create_app()

