"""DRISHTI - Mocked Depth & Spatial Photography Pipeline Integration Tests
Verifies the end-to-end integration of detection, segmentation, depth estimation,
and spatial photo representation using deterministic mock adapters.
"""

import io
from pathlib import Path
from PIL import Image
import pytest

from app.assets.manager import asset_manager
from app.events.emitter import event_emitter
from app.jobs.tracker import job_tracker
from app.pipeline.coordinator import PipelineCoordinator
from app.pipeline.mock import (
    MockDepthEstimator,
    MockPersonDetector,
    MockPersonSegmentor,
)
from app.scene.manager import scene_manager


@pytest.fixture
def mock_scene_setup(tmp_path: Path):
    """Creates a temporary test photo and associated scene."""
    img = Image.new("RGB", (320, 240), color=(120, 160, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    content = buf.getvalue()

    photo_asset = asset_manager.register_photo(
        content=content,
        filename="test_spatial_scene.jpg",
        mime_type="image/jpeg",
    )
    scene = scene_manager.create_scene_from_photo(
        photo_asset=photo_asset,
        name="Test Spatial Photo Scene",
    )
    return scene, photo_asset


@pytest.mark.asyncio
async def test_mocked_pipeline_full_spatial_photo(mock_scene_setup):
    """Verifies that the full pipeline generates detection, segmentation, depth, and spatial-photo assets."""
    scene, photo_asset = mock_scene_setup
    job_id = job_tracker.create_job(scene.id)

    coordinator = PipelineCoordinator(
        detector=MockPersonDetector(simulated_people_count=2),
        segmentor=MockPersonSegmentor(),
        depth_estimator=MockDepthEstimator(),
    )

    updated_scene = await coordinator.execute_pipeline(
        scene_id=scene.id,
        job_id=job_id,
        include_depth=True,
    )

    # 1. Verification of scene status
    assert updated_scene.processing_status == "ready"
    assert len(updated_scene.entities) == 2

    # 2. Verification of person depth association
    for person in updated_scene.entities:
        assert person.depth is not None
        assert person.depth.status == "ready"
        assert 0.0 <= person.depth.median_proximity <= 1.0
        assert person.depth.method == "segmentation_mask"
        assert person.depth.depth_asset_id is not None

    # 3. Verification of scene depth metadata
    assert updated_scene.depth is not None
    assert updated_scene.depth.status == "ready"
    assert updated_scene.depth.model == "mock-depth-v1"
    assert updated_scene.depth.depth_convention == "relative_proximity"
    assert updated_scene.depth.normalization is not None
    assert updated_scene.depth.depth_asset_id is not None
    assert updated_scene.depth.visualization_asset_id is not None

    # 4. Verification of spatial photo representation
    spatial_rep = updated_scene.representations.spatial_photo
    assert spatial_rep.status == "ready"
    assert spatial_rep.uri is not None
    assert spatial_rep.config is not None
    assert spatial_rep.config.default_depth_strength == 0.08

    # 5. Verification of emitted event sequence
    events = event_emitter.get_events(job_id=job_id, since_sequence=0)
    assert len(events) >= 12
    stage_names = [e.stage_identifier for e in events]
    assert "image_validation" in stage_names
    assert "person_detection" in stage_names
    assert "instance_segmentation" in stage_names
    assert "depth_estimation" in stage_names
    assert "depth_normalization" in stage_names
    assert "depth_association" in stage_names
    assert "spatial_photo_preparation" in stage_names
    assert "completed" in stage_names


@pytest.mark.asyncio
async def test_mocked_pipeline_depth_failure_resilience(mock_scene_setup):
    """Verifies that a failure in depth estimation does NOT destroy detection and segmentation results."""
    scene, photo_asset = mock_scene_setup
    job_id = job_tracker.create_job(scene.id)

    coordinator = PipelineCoordinator(
        detector=MockPersonDetector(simulated_people_count=2),
        segmentor=MockPersonSegmentor(),
        depth_estimator=MockDepthEstimator(simulate_failure=True),
    )

    updated_scene = await coordinator.execute_pipeline(
        scene_id=scene.id,
        job_id=job_id,
        include_depth=True,
    )

    # Detections and segmentations MUST be preserved
    assert len(updated_scene.entities) == 2
    for person in updated_scene.entities:
        assert person.detection is not None
        assert person.detection.status == "ready"
        assert person.segmentation is not None
        assert person.segmentation.status == "ready"

    # Depth is honestly marked as failed
    assert updated_scene.depth is not None
    assert updated_scene.depth.status == "failed"
    assert "Simulated depth estimation failure" in str(updated_scene.depth.error)
    assert updated_scene.representations.spatial_photo.status == "failed"
