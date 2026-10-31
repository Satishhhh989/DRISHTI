"""DRISHTI - Pipeline & Inference API Endpoints
"""

import asyncio
from typing import Any, Dict, Optional
from fastapi import APIRouter, BackgroundTasks, Query
from pydantic import BaseModel, Field

from app.jobs.tracker import job_tracker
from app.pipeline.coordinator import pipeline_coordinator, PipelineCoordinator
from app.pipeline.detector import YoloPersonDetector
from app.pipeline.device import device_manager
from app.pipeline.mock import MockDepthEstimator, MockPersonDetector, MockPersonSegmentor
from app.pipeline.models import model_manager
from app.pipeline.segmentor import YoloPersonSegmentor
from app.scene.manager import scene_manager

router = APIRouter(prefix="/pipeline", tags=["Pipeline & Inference"])


class AnalyzeRequest(BaseModel):
    scene_id: str
    confidence_threshold: float = Field(default=0.35, ge=0.05, le=0.95)
    use_mock: bool = False
    simulate_zero_people: bool = False
    include_depth: bool = True
    quality_preset: str = Field(default="high", pattern="^(preview|high|maximum)$")


class AnalyzeResponse(BaseModel):
    job_id: str
    scene_id: str
    status: str
    message: str


class DownloadModelRequest(BaseModel):
    model_key: str = "yolov8n-seg"


@router.get("/hardware")
async def get_hardware_info() -> Dict[str, Any]:
    """Returns detected inference hardware (Apple Silicon MPS, CUDA, CPU)."""
    return device_manager.get_device_info()


@router.get("/models")
async def get_models_status() -> Dict[str, Any]:
    """Returns availability and paths of local model weights."""
    return {
        "models": model_manager.list_models(),
        "models_dir": str(model_manager.models_dir),
    }


@router.post("/models/download")
async def download_model(request: DownloadModelRequest) -> Dict[str, Any]:
    """Downloads model weights to data/models/."""
    path = model_manager.download_model(request.model_key)
    return {
        "model_key": request.model_key,
        "path": str(path),
        "status": "ready",
        "size_bytes": path.stat().st_size,
    }


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_scene(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
) -> AnalyzeResponse:
    """Queues a full Phase 02 analysis job (detection + segmentation) for a scene."""
    # Validate scene exists
    scene = scene_manager.get_scene(request.scene_id)
    job_id = job_tracker.create_job(scene.id)
    scene.active_job_id = job_id
    scene.processing_status = "processing"
    scene_manager.update_scene(scene)

    # Determine coordinator configuration
    if request.use_mock:
        count = 0 if request.simulate_zero_people else 2
        coord = PipelineCoordinator(
            detector=MockPersonDetector(simulated_people_count=count),
            segmentor=MockPersonSegmentor(),
            depth_estimator=MockDepthEstimator(),
        )
    else:
        coord = pipeline_coordinator

    # Run in background task to not block the request
    async def run_pipeline():
        try:
            await coord.execute_pipeline(
                scene_id=scene.id,
                job_id=job_id,
                confidence_threshold=request.confidence_threshold,
                include_depth=request.include_depth,
                quality_preset=request.quality_preset,
            )
        except Exception:
            # Errors are already recorded in event emitter and scene status
            pass

    background_tasks.add_task(run_pipeline)

    return AnalyzeResponse(
        job_id=job_id,
        scene_id=scene.id,
        status="running",
        message="Image intelligence & spatial photography pipeline started.",
    )
