"""DRISHTI - Job & Event Stream Endpoints
Version 1.2.0 (Phase 04: Progressive Real-Time Streaming & Cancellation)
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Header, Query, Request
from fastapi.responses import StreamingResponse
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


class CancelJobResponse(BaseModel):
    job_id: str
    status: str
    message: str


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
        current_stage=job.get("current_stage", "image_validation"),
        latest_sequence_number=latest_seq,
    )


@router.get("/{job_id}/events", response_model=JobEventsResponse)
async def get_job_events(
    job_id: str,
    since: int = Query(default=0, ge=0, description="Retrieve events with sequence number strictly greater than this value"),
) -> JobEventsResponse:
    job = job_tracker.get_job(job_id)
    if not job:
        raise NotFoundError("Job", job_id)
    events = event_emitter.get_events(job_id=job_id, since_sequence=since)
    latest_seq = event_emitter.get_latest_sequence(job_id=job_id)
    return JobEventsResponse(
        job_id=job_id,
        events=events,
        latest_sequence_number=latest_seq,
    )


@router.post("/{job_id}/cancel", response_model=CancelJobResponse)
async def cancel_job(job_id: str) -> CancelJobResponse:
    """Cancels an active or pending processing job."""
    job = job_tracker.get_job(job_id)
    if not job:
        raise NotFoundError("Job", job_id)

    cancelled = job_tracker.cancel_job(job_id)
    if cancelled:
        return CancelJobResponse(
            job_id=job_id,
            status="cancelled",
            message="Job successfully cancelled.",
        )
    return CancelJobResponse(
        job_id=job_id,
        status=job.get("status", "cancelled"),
        message=f"Job was already in terminal state '{job.get('status')}'.",
    )


@router.get("/{job_id}/stream")
async def stream_job_events(
    job_id: str,
    request: Request,
    since: int = Query(default=0, ge=0, description="Sequence number to resume from"),
    last_event_id: Optional[str] = Header(default=None, alias="Last-Event-ID"),
):
    """Streams processing events as Server-Sent Events (SSE) with catch-up and replay."""
    job = job_tracker.get_job(job_id)
    if not job:
        raise NotFoundError("Job", job_id)

    # Use Last-Event-ID if provided and higher than since
    resume_seq = since
    if last_event_id and last_event_id.isdigit():
        resume_seq = max(resume_seq, int(last_event_id))

    async def sse_event_generator():
        try:
            async for event in event_emitter.subscribe(job_id=job_id, since_sequence=resume_seq):
                if await request.is_disconnected():
                    break
                payload = event.model_dump_json()
                yield f"id: {event.sequence_number}\nevent: {event.event_type}\ndata: {payload}\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
