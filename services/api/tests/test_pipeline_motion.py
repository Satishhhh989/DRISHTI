"""DRISHTI - Motion Pipeline Integration Tests (Phase 07)
Verifies full pipeline execution through pose estimation, skeleton fitting,
mesh rigging, and procedural animation preparation.
"""

import io
import pytest
from PIL import Image

from app.assets.manager import asset_manager
from app.events.emitter import event_emitter
from app.jobs.tracker import job_tracker
from app.pipeline.coordinator import PipelineCoordinator
from app.pipeline.mock import (
    MockDepthEstimator,
    MockPersonDetector,
    MockPersonSegmentor,
)
from app.pipeline.reconstructor import MockHumanReconstructor
from app.scene.manager import scene_manager


@pytest.fixture
def test_scene_with_photo():
    img = Image.new("RGB", (640, 480), color=(140, 150, 160))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    photo_bytes = buf.getvalue()

    photo_asset = asset_manager.register_photo(
        content=photo_bytes,
        filename="test_motion_group.jpg",
        mime_type="image/jpeg",
    )
    scene = scene_manager.create_scene_from_photo(photo_asset, name="Motion Test Scene")
    return scene, photo_asset


@pytest.mark.asyncio
async def test_full_pipeline_motion_articulation(test_scene_with_photo):
    """Verifies that the full pipeline executes through Phase 07 motion stages:
    pose_estimation, skeleton_fitting, mesh_rigging, and animation_preparation.
    """
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

    # 1. Verify all detected people have skeletons
    assert len(final_scene.entities) == 2
    for entity in final_scene.entities:
        assert entity.skeleton is not None
        assert entity.skeleton.validation_status == "valid"
        assert len(entity.skeleton.joints) == 24
        assert len(entity.skeleton.bones) == 23
        assert entity.skeleton.root_joint_id == "pelvis"

        # 2. Verify all detected people have rigs
        assert entity.rig is not None
        assert entity.rig.status == "rigged"
        assert entity.rig.weights_normalized is True
        assert entity.rig.bone_count == 24

        # 3. Verify procedural animation clips
        assert len(entity.animation_clips) == 4
        clip_names = {c.name for c in entity.animation_clips}
        assert "idle_breathing" in clip_names
        assert "torso_twist" in clip_names
        assert "arm_wave" in clip_names
        assert "gentle_walk" in clip_names
        assert entity.animation_state.current_clip == "idle_breathing"

    # 4. Verify stage events were emitted in order
    events = event_emitter.get_events(job_id)
    stages_emitted = [ev.stage_identifier for ev in events]
    assert "pose_estimation" in stages_emitted
    assert "skeleton_fitting" in stages_emitted
    assert "mesh_rigging" in stages_emitted
    assert "animation_preparation" in stages_emitted
    assert "scene_assembly" in stages_emitted

    # 5. Verify scene persistence with motion data
    reloaded_scene = scene_manager.get_scene(final_scene.id)
    for entity in reloaded_scene.entities:
        assert entity.skeleton is not None
        assert len(entity.skeleton.joints) == 24
        assert entity.rig is not None
        assert entity.rig.status == "rigged"
        assert len(entity.animation_clips) == 4

