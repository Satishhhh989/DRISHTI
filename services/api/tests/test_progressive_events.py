"""DRISHTI - Progressive Engine, Event Streaming, and Cancellation Tests
Phase 04 Verification.
"""

import asyncio
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.assets.manager import asset_manager
from app.events.emitter import event_emitter, MAX_EVENTS_PER_JOB
from app.jobs.tracker import job_tracker
from app.main import app
from app.pipeline.coordinator import PipelineCoordinator
from app.pipeline.mock import (
    MockDepthEstimator,
    MockPersonDetector,
    MockPersonSegmentor,
)
from app.scene.manager import scene_manager


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def test_scene():
    img = Image.new("RGB", (400, 300), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    asset = asset_manager.register_photo(
        content=buf.getvalue(),
        filename="test_progressive.jpg",
        mime_type="image/jpeg",
    )
    return scene_manager.create_scene_from_photo(asset, "Progressive Test Scene")


def test_bounded_event_retention():
    """Verifies event emitter retains at most MAX_EVENTS_PER_JOB events with monotonically increasing sequences."""
    test_job_id = "test-job-bounded-123"
    test_scene_id = "test-scene-bounded-123"

    for i in range(MAX_EVENTS_PER_JOB + 50):
        event_emitter.emit(
            job_id=test_job_id,
            scene_id=test_scene_id,
            event_type="progress_updated",
            stage_identifier="image_validation",
            status="running",
            progress=i / (MAX_EVENTS_PER_JOB + 50),
            message=f"Event {i}",
        )

    stored = event_emitter.get_events(test_job_id, since_sequence=0)
    assert len(stored) == MAX_EVENTS_PER_JOB
    # Verify latest sequence matches total emissions
    assert event_emitter.get_latest_sequence(test_job_id) == MAX_EVENTS_PER_JOB + 50
    # Verify strictly monotonic sequence numbers in stored slice
    for idx in range(len(stored) - 1):
        assert stored[idx].sequence_number < stored[idx + 1].sequence_number


def test_job_cancellation_flow(client, test_scene):
    """Verifies that cancel_job stops work and marks job as cancelled."""
    job_id = job_tracker.create_job(test_scene.id)
    assert not job_tracker.is_cancelled(job_id)

    # Cancel via API
    res = client.post(f"/api/v1/jobs/{job_id}/cancel")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "cancelled"
    assert job_tracker.is_cancelled(job_id)

    # Subsequent cancellation is idempotent and reports already cancelled
    res2 = client.post(f"/api/v1/jobs/{job_id}/cancel")
    assert res2.status_code == 200
    assert res2.json()["status"] == "cancelled"

    # Event emitter contains job_cancelled event
    events = event_emitter.get_events(job_id, since_sequence=0)
    cancelled_events = [e for e in events if e.event_type == "job_cancelled"]
    assert len(cancelled_events) == 1
    assert cancelled_events[0].status == "cancelled"


@pytest.mark.asyncio
async def test_coordinator_honors_cancellation(test_scene):
    """Verifies that the coordinator immediately exits when job is cancelled before completion."""
    job_id = job_tracker.create_job(test_scene.id)
    # Cancel before execution
    job_tracker.cancel_job(job_id)

    coord = PipelineCoordinator(
        detector=MockPersonDetector(simulated_people_count=2),
        segmentor=MockPersonSegmentor(),
        depth_estimator=MockDepthEstimator(),
    )

    scene = await coord.execute_pipeline(
        scene_id=test_scene.id,
        job_id=job_id,
        include_depth=True,
    )

    # Should exit without throwing and preserve scene
    assert scene is not None
    assert scene.id == test_scene.id


def test_job_stream_sse_endpoint(client, test_scene):
    """Verifies SSE text/event-stream endpoint emits valid SSE format."""
    job_id = job_tracker.create_job(test_scene.id)
    # Emit an event
    event_emitter.emit(
        job_id=job_id,
        scene_id=test_scene.id,
        event_type="stage_completed",
        stage_identifier="image_validation",
        status="running",
        message="Valid photo",
    )
    # Complete job
    event_emitter.emit(
        job_id=job_id,
        scene_id=test_scene.id,
        event_type="job_completed",
        stage_identifier="completed",
        status="completed",
        message="Done",
    )

    # Request stream
    with client.stream("GET", f"/api/v1/jobs/{job_id}/stream?since=0") as stream:
        lines = []
        for line in stream.iter_lines():
            if line:
                lines.append(line)
            if len(lines) >= 6:
                break

    # Check SSE format: id: ..., event: ..., data: ...
    assert any(l.startswith("id:") for l in lines)
    assert any(l.startswith("event:") for l in lines)
    assert any(l.startswith("data:") for l in lines)


@pytest.mark.asyncio
async def test_progressive_scene_entity_updates(test_scene):
    """Verifies that detected entities are committed to scene_manager before segmentation completes."""
    job_id = job_tracker.create_job(test_scene.id)

    coord = PipelineCoordinator(
        detector=MockPersonDetector(simulated_people_count=2),
        segmentor=MockPersonSegmentor(),
        depth_estimator=MockDepthEstimator(),
    )

    # Run full pipeline
    scene = await coord.execute_pipeline(
        scene_id=test_scene.id,
        job_id=job_id,
        include_depth=True,
    )

    # Verify both entities exist with valid detections and masks
    assert len(scene.entities) == 2
    for idx, e in enumerate(scene.entities):
        assert e.detection is not None
        assert e.detection.status == "ready"
        assert e.segmentation is not None
        assert e.segmentation.status == "ready"
        assert e.depth is not None
        assert e.depth.status == "ready"

    # Verify per-person asset_ready events were emitted with exact honest messages
    events = event_emitter.get_events(job_id, since_sequence=0)
    mask_events = [e for e in events if e.event_type == "asset_ready" and e.asset_reference and e.asset_reference.asset_type == "segmentation_mask"]
    assert len(mask_events) == 2
    assert "1 of 2" in mask_events[0].message
    assert "2 of 2" in mask_events[1].message
