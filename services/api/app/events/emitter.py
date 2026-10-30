"""PHOTO → 3D ENGINE - Processing Event Emitter

Publishes typed, strictly-sequenced processing events and provides
reconciliation capabilities for reconnecting clients.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional
from app.core.logging import logger
from app.schemas.events import (
    AssetReference,
    JobStatus,
    PipelineStageIdentifier,
    ProcessingEvent,
    ProcessingEventType,
    StructuredError,
)


class EventEmitter:
    def __init__(self):
        # job_id -> list of ProcessingEvent
        self._event_store: Dict[str, List[ProcessingEvent]] = {}
        # job_id -> sequence counter
        self._sequences: Dict[str, int] = {}

    def emit(
        self,
        job_id: str,
        scene_id: str,
        event_type: ProcessingEventType,
        stage_identifier: PipelineStageIdentifier,
        status: JobStatus,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        entity_id: Optional[str] = None,
        asset_reference: Optional[AssetReference] = None,
        error: Optional[StructuredError] = None,
    ) -> ProcessingEvent:
        """Creates and stores a sequenced event for a job."""
        current_seq = self._sequences.get(job_id, 0) + 1
        self._sequences[job_id] = current_seq

        event = ProcessingEvent(
            event_id=str(uuid.uuid4()),
            job_id=job_id,
            scene_id=scene_id,
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            sequence_number=current_seq,
            stage_identifier=stage_identifier,
            status=status,
            progress=progress,
            message=message,
            entity_id=entity_id,
            asset_reference=asset_reference,
            error=error,
        )

        if job_id not in self._event_store:
            self._event_store[job_id] = []
        self._event_store[job_id].append(event)

        logger.info(
            f"Event [{current_seq}] job={job_id} stage={stage_identifier} type={event_type} status={status}"
        )
        return event

    def get_events(
        self,
        job_id: str,
        since_sequence: int = 0
    ) -> List[ProcessingEvent]:
        """Returns all events for a job strictly greater than since_sequence."""
        events = self._event_store.get(job_id, [])
        return [e for e in events if e.sequence_number > since_sequence]

    def get_latest_sequence(self, job_id: str) -> int:
        return self._sequences.get(job_id, 0)


event_emitter = EventEmitter()

