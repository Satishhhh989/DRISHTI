"""PHOTO → 3D ENGINE - Event Stream & Catch-up Endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.events.emitter import event_emitter
from app.schemas.events import ProcessingEvent

router = APIRouter(prefix="/jobs", tags=["Jobs & Events"])


class JobEventsResponse(BaseModel):
    job_id: str
    events: List[ProcessingEvent]
    latest_sequence_number: int


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
