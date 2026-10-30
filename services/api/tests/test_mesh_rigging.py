"""DRISHTI - Mesh Rigging & Skinning Tests (Phase 07)
"""

from pathlib import Path
import numpy as np
import pytest

from app.pipeline.motion import (
    RiggingEngine,
    SkeletonFitter,
    validate_skinning_weights,
)
from app.schemas.common import BoundingBox2D
from app.schemas.scene import HumanReconstructionData, SceneEntity


def test_skinning_weight_validation_passes_normalized():
    """Verifies that normalized unit-sum skinning weights pass validation."""
    num_verts = 50
    # Random weights normalized to 1.0
    w = np.random.uniform(0.1, 1.0, (num_verts, 4))
    w = w / np.sum(w, axis=1, keepdims=True)
    indices = np.random.randint(0, 24, (num_verts, 4))

    is_valid, issues = validate_skinning_weights(w, indices, bone_count=24)
    assert is_valid
    assert len(issues) == 0


def test_skinning_weight_validation_rejects_unnormalized():
    """Verifies that non-unit sum skinning weights fail validation."""
    w = np.ones((10, 4)) * 0.5  # sums to 2.0
    indices = np.zeros((10, 4), dtype=int)

    is_valid, issues = validate_skinning_weights(w, indices, bone_count=24)
    assert not is_valid
    assert any("do not sum to 1.0" in msg for msg in issues)


def test_skinning_weight_validation_rejects_out_of_range_indices():
    """Verifies that bone indices outside [0, bone_count-1] fail validation."""
    w = np.ones((5, 4)) * 0.25
    indices = np.zeros((5, 4), dtype=int)
    indices[0, 0] = 99  # Invalid bone index

    is_valid, issues = validate_skinning_weights(w, indices, bone_count=24)
    assert not is_valid
    assert any("Bone indices out of valid range" in msg for msg in issues)


def test_proximity_heat_rigging_engine():
    """Verifies proximity heat rigging generates normalized top-4 weights for mock mesh geometry."""
    fitter = SkeletonFitter()
    rig_engine = RiggingEngine()

    entity = SceneEntity(
        id="person-001",
        source_image_id="photo-1",
        bounding_box=BoundingBox2D(x_min=0.2, y_min=0.1, x_max=0.5, y_max=0.9),
    )

    mock_verts = np.random.uniform(-0.5, 0.5, (200, 3))
    skeleton = fitter.build_skeleton(entity, mock_verts)
    assert skeleton.validation_status == "valid"

    rig = rig_engine.rig_mesh(entity, skeleton, mock_verts)
    assert rig.status == "rigged"
    assert rig.skinning_method == "proximity_heat_weights"
    assert rig.max_influences_per_vertex == 4
    assert rig.weights_normalized is True
    assert rig.bone_count == 24


def test_unrigged_handling_when_mesh_missing():
    """Verifies that missing mesh geometry produces unrigged status cleanly without corrupting entity."""
    fitter = SkeletonFitter()
    rig_engine = RiggingEngine()

    entity = SceneEntity(
        id="person-002",
        source_image_id="photo-1",
    )
    skeleton = fitter.build_skeleton(entity, None)

    rig = rig_engine.rig_mesh(entity, skeleton, None)
    assert rig.status == "unrigged"
    assert "No mesh geometry available" in rig.error_message


def test_smpl_native_rigging_if_model_present():
    """Verifies SMPL native skinning weights when SMPL_N_model.npz is available."""
    smpl_path = Path("data/models/human_reconstruction/SMPL_N_model.npz")
    if not smpl_path.exists():
        pytest.skip("SMPL_N_model.npz not found; skipping native SMPL rigging test.")

    rig_engine = RiggingEngine(smpl_npz_path=smpl_path)
    fitter = SkeletonFitter(smpl_npz_path=smpl_path)

    entity = SceneEntity(
        id="person-smpl",
        source_image_id="photo-1",
    )

    # 6,890 SMPL template vertices
    npz = np.load(smpl_path, allow_pickle=True)
    v_template = npz["v_template"].astype(np.float32)

    skeleton = fitter.build_skeleton(entity, v_template)
    assert skeleton.source == "smpl_regressed"

    rig = rig_engine.rig_mesh(entity, skeleton, v_template)
    assert rig.status == "rigged"
    assert rig.skinning_method == "smpl_lbs_native"
    assert rig.weights_normalized is True
