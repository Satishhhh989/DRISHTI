"""DRISHTI - Processing Event Emitter
Version 1.2.0 (Phase 04: Progressive Engine & Real-Time Event Stream)

Publishes typed, strictly-sequenced processing events, maintains bounded retention,
and provides live SSE pub/sub streaming and reconciliation for reconnecting clients.
"""

import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, List, Optional, Set
import uuid

from app.core.logging import logger
from app.schemas.events import (
    AssetReference,
    JobStatus,
    PipelineStageIdentifier,
    ProcessingEvent,
    ProcessingEventType,
    StructuredError,
)

MAX_EVENTS_PER_JOB = 500


class EventEmitter:
    def __init__(self):
        # job_id -> list of ProcessingEvent (bounded FIFO)
        self._event_store: Dict[str, List[ProcessingEvent]] = {}
        # job_id -> sequence counter
        self._sequences: Dict[str, int] = {}
        # job_id -> set of active subscriber queues
        self._subscribers: Dict[str, Set[asyncio.Queue]] = {}

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
        """Creates, stores, and broadcasts a sequenced event for a job."""
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
        
        # Bounded retention: prevent unbounded memory consumption
        store = self._event_store[job_id]
        store.append(event)
        if len(store) > MAX_EVENTS_PER_JOB:
            # Drop oldest events beyond boundary
            self._event_store[job_id] = store[-MAX_EVENTS_PER_JOB:]

        logger.info(
            f"Event [{current_seq}] job={job_id} stage={stage_identifier} type={event_type} status={status}"
        )

        # Broadcast event to all live SSE subscriber queues
        subscribers = self._subscribers.get(job_id, set())
        for q in list(subscribers):
            try:
                q.put_nowait(event)
            except Exception as e:
                logger.debug(f"Failed to put event into subscriber queue: {e}")

        return event

    def get_events(
        self,
        job_id: str,
        since_sequence: int = 0
    ) -> List[ProcessingEvent]:
        """Returns all stored events for a job strictly greater than since_sequence."""
        events = self._event_store.get(job_id, [])
        return [e for e in events if e.sequence_number > since_sequence]

    def get_latest_sequence(self, job_id: str) -> int:
        return self._sequences.get(job_id, 0)

    async def subscribe(
        self,
        job_id: str,
        since_sequence: int = 0,
        keepalive_timeout: float = 15.0,
    ) -> AsyncGenerator[ProcessingEvent, None]:
        """Yields past events since since_sequence, then yields live incoming events until completion."""
        queue: asyncio.Queue = asyncio.Queue()

        if job_id not in self._subscribers:
            self._subscribers[job_id] = set()
        self._subscribers[job_id].add(queue)

        try:
            # 1. Replay historical events since requested sequence
            past_events = self.get_events(job_id, since_sequence=since_sequence)
            last_yielded_seq = since_sequence
            for event in past_events:
                last_yielded_seq = max(last_yielded_seq, event.sequence_number)
                yield event
                if event.event_type in ("job_completed", "job_cancelled"):
                    return

            # 2. Stream live incoming events from queue
            while True:
                try:
                    event: ProcessingEvent = await asyncio.wait_for(
                        queue.get(),
                        timeout=keepalive_timeout,
                    )
                except asyncio.TimeoutError:
                    # Timeout reached; continue loop to allow ping/heartbeat handling by consumer
                    continue

                if event.sequence_number > last_yielded_seq:
                    last_yielded_seq = event.sequence_number
                    yield event

                    if event.event_type in ("job_completed", "job_cancelled"):
                        break
                    if event.event_type == "stage_failed" and event.status in ("failed", "cancelled"):
                        break
        finally:
            if job_id in self._subscribers and queue in self._subscribers[job_id]:
                self._subscribers[job_id].remove(queue)
                if not self._subscribers[job_id]:
                    del self._subscribers[job_id]


event_emitter = EventEmitter()
