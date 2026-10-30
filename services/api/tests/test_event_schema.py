from app.events.emitter import EventEmitter
from app.schemas.events import ProcessingEvent, AssetReference


def test_event_sequencing():
    emitter = EventEmitter()
    job_id = "job-seq-test"
    scene_id = "scene-test"

    e1 = emitter.emit(
        job_id=job_id,
        scene_id=scene_id,
        event_type="job_accepted",
        stage_identifier="image_analysis",
        status="pending",
    )
    assert e1.sequence_number == 1

    e2 = emitter.emit(
        job_id=job_id,
        scene_id=scene_id,
        event_type="stage_started",
        stage_identifier="image_analysis",
        status="running",
    )
    assert e2.sequence_number == 2

    # Query since sequence 1 should only return e2
    events_after_1 = emitter.get_events(job_id=job_id, since_sequence=1)
    assert len(events_after_1) == 1
    assert events_after_1[0].sequence_number == 2

    # Query since sequence 0 should return both
    all_events = emitter.get_events(job_id=job_id, since_sequence=0)
    assert len(all_events) == 2


def test_event_with_asset_reference():
    emitter = EventEmitter()
    asset_ref = AssetReference(
        asset_id="asset-mask-01",
        asset_type="segmentation_mask",
        uri="/api/v1/assets/asset-mask-01/file",
        mime_type="image/png",
    )
    evt = emitter.emit(
        job_id="job-asset-test",
        scene_id="scene-asset-test",
        event_type="asset_ready",
        stage_identifier="segmentation",
        status="running",
        asset_reference=asset_ref,
    )
    assert evt.asset_reference is not None
    assert evt.asset_reference.asset_type == "segmentation_mask"

