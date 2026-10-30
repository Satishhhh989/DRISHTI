import io
import pytest
from PIL import Image
from app.assets.manager import asset_manager
from app.events.emitter import event_emitter
from app.pipeline.coordinator import PipelineCoordinator
from app.pipeline.mock import MockPersonDetector, MockPersonSegmentor
from app.scene.manager import scene_manager


@pytest.mark.asyncio
async def test_mocked_pipeline_full_run():
    # 1. Register a test photograph
    img = Image.new("RGB", (800, 600), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    photo_asset = asset_manager.register_photo(
        content=buf.getvalue(),
        filename="test_mock_photo.jpg",
        mime_type="image/jpeg",
    )

    # 2. Create scene
    scene = scene_manager.create_scene_from_photo(photo_asset, name="Test Mock Scene")

    # 3. Setup coordinator with deterministic mock detector (2 people) and mock segmentor
    coord = PipelineCoordinator(
        detector=MockPersonDetector(simulated_people_count=2),
        segmentor=MockPersonSegmentor(simulate_partial_failure=False),
    )

    job_id = "job-mock-test-01"
    updated_scene = await coord.execute_phase02_pipeline(
        scene_id=scene.id,
        job_id=job_id,
        confidence_threshold=0.35,
    )

    # Assert scene state
    assert updated_scene.processing_status == "ready"
    assert len(updated_scene.entities) == 2

    # Verify Person 1
    p1 = updated_scene.entities[0]
    assert p1.detection is not None
    assert p1.detection.confidence == 0.92
    assert p1.segmentation is not None
    assert p1.segmentation.status == "ready"
    assert p1.segmentation.mask_asset_id is not None
    assert p1.reconstruction_status == "unstarted"  # 3D reconstruction is Phase 06

    # Verify Person 2
    p2 = updated_scene.entities[1]
    assert p2.detection is not None
    assert p2.segmentation is not None
    assert p2.segmentation.status == "ready"

    # Verify mask asset persistence and retrieval
    mask_asset = asset_manager.get_asset(p1.segmentation.mask_asset_id)
    assert mask_asset.category == "segmentation_mask"
    mask_path = asset_manager.get_asset_path(mask_asset.asset_id)
    assert mask_path.exists()

    # Reopen mask asset and verify dimensions match original image
    with Image.open(mask_path) as mask_img:
        assert mask_img.size == (800, 600)

    # Verify event sequencing
    events = event_emitter.get_events(job_id=job_id, since_sequence=0)
    assert len(events) >= 8
    # Assert sequence numbers are strictly monotonically increasing: 1, 2, 3, ...
    for i in range(len(events) - 1):
        assert events[i].sequence_number < events[i + 1].sequence_number


@pytest.mark.asyncio
async def test_mocked_pipeline_zero_people():
    # Test zero-person legitimate result
    img = Image.new("RGB", (640, 480), color=(50, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    photo_asset = asset_manager.register_photo(
        content=buf.getvalue(),
        filename="empty_landscape.jpg",
    )
    scene = scene_manager.create_scene_from_photo(photo_asset, name="Empty Landscape")

    coord = PipelineCoordinator(
        detector=MockPersonDetector(simulated_people_count=0),
        segmentor=MockPersonSegmentor(),
    )

    job_id = "job-zero-people"
    updated_scene = await coord.execute_phase02_pipeline(
        scene_id=scene.id,
        job_id=job_id,
    )

    assert updated_scene.processing_status == "ready"
    assert len(updated_scene.entities) == 0


@pytest.mark.asyncio
async def test_mocked_pipeline_partial_segmentation_failure():
    # Test when 1 person's segmentation fails, detection is preserved
    img = Image.new("RGB", (800, 600), color=(80, 80, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    photo_asset = asset_manager.register_photo(
        content=buf.getvalue(),
        filename="two_people_partial.jpg",
    )
    scene = scene_manager.create_scene_from_photo(photo_asset, name="Partial Failure Scene")

    coord = PipelineCoordinator(
        detector=MockPersonDetector(simulated_people_count=2),
        segmentor=MockPersonSegmentor(simulate_partial_failure=True),
    )

    job_id = "job-partial-fail"
    updated_scene = await coord.execute_phase02_pipeline(
        scene_id=scene.id,
        job_id=job_id,
    )

    assert len(updated_scene.entities) == 2
    # First person has mask
    assert updated_scene.entities[0].segmentation.status == "ready"
    # Second person has detection preserved, but segmentation failed
    assert updated_scene.entities[1].detection is not None
    assert updated_scene.entities[1].segmentation.status == "failed"
