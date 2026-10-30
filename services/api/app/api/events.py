"""DRISHTI - Job & Event Stream Endpoints
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.core.errors import NotFoundError
from app.events.emitter import event_emitter
from app.jobs.tracker import job_tracker
from app.schemas.events import ProcessingEvent

router = APIRouter(prefix="/jobs", tags=["Jobs & Events"])


class JobEventsResponse(BaseModel):
    job_id: str
    events: List[ProcessingEvent]
    latest_sequence_number: int


class JobStatusResponse(BaseModel):
    job_id: str
    scene_id: str
    status: str
    current_stage: str
    latest_sequence_number: int


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    job = job_tracker.get_job(job_id)
    if not job:
        raise NotFoundError("Job", job_id)
    latest_seq = event_emitter.get_latest_sequence(job_id=job_id)
    return JobStatusResponse(
        job_id=job["job_id"],
        scene_id=job["scene_id"],
        status=job.get("status", "pending"),
        current_stage=job.get("current_stage", "image_analysis"),
        latest_sequence_number=latest_seq,
    )


@router.get("/{job_id}/events", response_model=JobEventsResponse)
async def get_job_events(
    job_id: str,
    since: int = Query(default=0, ge=0, description="Retrieve events with sequence number strictly greater than this value")
) -> JobEventsResponse:
    events = event_emitter.get_events(job_id=job_id, since_sequence=since)
    latest_seq = event_emitter.get_latest_sequence(job_id=job_id)
    return JobEventsResponse(
        job_id=job_id,
        events=events,
        latest_sequence_number=latest_seq
    )
