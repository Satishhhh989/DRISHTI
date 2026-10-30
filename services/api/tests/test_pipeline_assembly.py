"""DRISHTI - Scene Assembly Pipeline Integration Tests (Phase 06)
Verifies full pipeline execution with deterministic multi-person scene assembly,
SceneGraph generation, and AssemblyReport attachment.
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
    img = Image.new("RGB", (600, 400), color=(150, 160, 170))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    photo_bytes = buf.getvalue()

    photo_asset = asset_manager.register_photo(
        content=photo_bytes,
        filename="test_assembly_group.jpg",
        mime_type="image/jpeg",
    )
    scene = scene_manager.create_scene_from_photo(photo_asset, name="Assembly Test Scene")
    return scene, photo_asset


@pytest.mark.asyncio
async def test_full_pipeline_scene_assembly(test_scene_with_photo):
    """Verifies that the full pipeline executes through stage 12 'scene_assembly' and constructs a valid SceneGraph."""
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

    # 1. Verify SceneGraph exists
    assert final_scene.scene_graph is not None
    assert final_scene.scene_graph.root_id == "scene-root"
    assert "scene-root" in final_scene.scene_graph.nodes
    assert "camera-main" in final_scene.scene_graph.nodes
    assert "plane-reference" in final_scene.scene_graph.nodes
    assert "plane-ground" in final_scene.scene_graph.nodes

    # 2. Verify all 2 detected entities are assembled
    assert len(final_scene.entities) == 2
    for entity in final_scene.entities:
        assert entity.placement is not None
        assert entity.placement.method == "depth_projected_composition"
        assert entity.scene_node_id is not None
        assert entity.scene_node_id in final_scene.scene_graph.nodes

        # Node transform matches entity transform
        node = final_scene.scene_graph.nodes[entity.scene_node_id]
        assert node.entity_id == entity.id
        assert node.transform.position.x == entity.transform.position.x
        assert node.transform.position.y == entity.transform.position.y
        assert node.transform.position.z == entity.transform.position.z

    # 3. Verify horizontal ordering: Person 1 (left) < Person 2 (right)
    assert final_scene.entities[0].transform.position.x < final_scene.entities[1].transform.position.x

    # 4. Verify SceneAssemblyReport
    assert final_scene.assembly_report is not None
    assert final_scene.assembly_report.total_entities == 2
    assert final_scene.assembly_report.reconstructed_count == 2
    assert final_scene.assembly_report.unreconstructed_count == 0
    assert final_scene.assembly_report.status == "completed"

    # 5. Verify events emitted
    events = event_emitter.get_events(job_id)
    stages_emitted = [ev.stage_identifier for ev in events]
    assert "scene_assembly" in stages_emitted
