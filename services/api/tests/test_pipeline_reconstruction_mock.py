"""DRISHTI - Mocked Human Reconstruction Pipeline Integration Tests (Phase 05)
"""

import io
import pytest
from PIL import Image
import numpy as np

from app.assets.manager import asset_manager
from app.events.emitter import event_emitter
from app.jobs.tracker import job_tracker
from app.pipeline.coordinator import PipelineCoordinator
from app.pipeline.mock import MockDepthEstimator, MockPersonDetector, MockPersonSegmentor
from app.pipeline.reconstructor import MockHumanReconstructor, HumanReconstructionResult
from app.scene.manager import scene_manager
from app.schemas.common import BoundingBox2D


@pytest.fixture
def test_scene_with_photo():
    img = Image.new("RGB", (400, 300), color=(180, 190, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    photo_bytes = buf.getvalue()

    photo_asset = asset_manager.register_photo(
        content=photo_bytes,
        filename="test_multi_person.jpg",
        mime_type="image/jpeg",
    )
    scene = scene_manager.create_scene_from_photo(photo_asset, name="Test Reconstruction Scene")
    return scene, photo_asset


@pytest.mark.asyncio
async def test_mocked_pipeline_full_human_reconstruction(test_scene_with_photo):
    """Verifies that the pipeline reconstructs all detected people into 3D meshes and enables representations."""
    scene, _ = test_scene_with_photo
    job_id = job_tracker.create_job(scene.id)

    coordinator = PipelineCoordinator(
        detector=MockPersonDetector(simulated_people_count=2),
        segmentor=MockPersonSegmentor(),
        depth_estimator=MockDepthEstimator(),
        reconstructor=MockHumanReconstructor(),
    )

    final_scene = await coordinator.execute_pipeline(
        scene_id=scene.id,
        job_id=job_id,
        confidence_threshold=0.3,
        include_depth=True,
        include_reconstruction=True,
    )

    # 1. Check entities have reconstruction data
    assert len(final_scene.entities) == 2
    for entity in final_scene.entities:
        assert entity.reconstruction is not None
        assert entity.reconstruction.status == "ready"
        assert entity.reconstruction.mesh_asset_id is not None
        assert entity.reconstruction.mesh_url is not None
        assert entity.reconstruction.geometry_metadata is not None
        assert entity.reconstruction.geometry_metadata.vertex_count >= 1000
        assert entity.reconstruction.geometry_metadata.face_count > 1500
        assert entity.geometry_ref == entity.reconstruction.mesh_asset_id

    # 2. Check scene representations enabled
    assert final_scene.representations.wireframe.status == "ready"
    assert final_scene.representations.solid3d.status == "ready"
    assert final_scene.representations.photo.status == "ready"
    assert final_scene.representations.spatial_photo.status == "ready"

    # 3. Check events emitted
    events = event_emitter.get_events(job_id)
    mesh_events = [e for e in events if e.stage_identifier == "human_reconstruction"]
    assert len(mesh_events) >= 3  # stage_started, 2 asset_ready, stage_completed
    asset_ready_events = [e for e in mesh_events if e.event_type == "asset_ready"]
    assert len(asset_ready_events) == 2
    for e in asset_ready_events:
        assert e.asset_reference is not None
        assert e.asset_reference.asset_type == "mesh"


class FailingPersonReconstructor(MockHumanReconstructor):
    """Fails reconstruction on the second person to test partial failure resilience."""
    def __init__(self):
        super().__init__()
        self.call_count = 0

    def reconstruct(self, image_rgb, bounding_box, mask_gray=None, depth_stats=None):
        self.call_count += 1
        if self.call_count == 2:
            raise RuntimeError("Simulated GPU out of memory during mesh generation.")
        return super().reconstruct(image_rgb, bounding_box, mask_gray, depth_stats)


@pytest.mark.asyncio
async def test_mocked_pipeline_partial_reconstruction_failure_isolation(test_scene_with_photo):
    """Verifies that if one person's reconstruction fails, successful people remain selectable and ready."""
    scene, _ = test_scene_with_photo
    job_id = job_tracker.create_job(scene.id)

    coordinator = PipelineCoordinator(
        detector=MockPersonDetector(simulated_people_count=2),
        segmentor=MockPersonSegmentor(),
        depth_estimator=MockDepthEstimator(),
        reconstructor=FailingPersonReconstructor(),
    )

    final_scene = await coordinator.execute_pipeline(
        scene_id=scene.id,
        job_id=job_id,
        confidence_threshold=0.3,
        include_depth=True,
        include_reconstruction=True,
    )

    assert len(final_scene.entities) == 2
    # Person 1 succeeded
    person_1 = final_scene.entities[0]
    assert person_1.reconstruction.status == "ready"
    assert person_1.reconstruction.mesh_url is not None

    # Person 2 failed
    person_2 = final_scene.entities[1]
    assert person_2.reconstruction.status == "failed"
    assert "out of memory" in person_2.reconstruction.error_message

    # Representations still enabled because at least 1 person is reconstructed
    assert final_scene.representations.wireframe.status == "ready"
    assert final_scene.representations.solid3d.status == "ready"
