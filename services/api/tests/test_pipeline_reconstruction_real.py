"""DRISHTI - Real Pretrained Model Smoke Tests (Phase 05)
"""

import io
from pathlib import Path
import numpy as np
import pytest
import torch
import trimesh

from app.pipeline.device import device_manager
from app.pipeline.models import model_manager
from app.pipeline.reconstructor import SpinHumanReconstructor
from app.schemas.common import BoundingBox2D


def test_real_spin_reconstruction_smoke():
    """Runs genuine local inference using the real SPIN checkpoint + SMPL Neutral LBS model.
    Skips cleanly if model weights are unavailable.
    """
    if not model_manager.is_model_available("spin-human-reconstruction"):
        pytest.skip("SPIN model weights not found in data/models/human_reconstruction; skipping real smoke test.")

    ckpt_path = model_manager.get_model_path("spin-human-reconstruction")
    smpl_path = ckpt_path.parent / "SMPL_N_model.npz"
    mean_path = ckpt_path.parent / "smpl_mean_params.npz"

    if not smpl_path.exists() or not mean_path.exists():
        pytest.skip("SMPL parameter assets not found; skipping real smoke test.")

    device = device_manager.device
    reconstructor = SpinHumanReconstructor(
        checkpoint_path=ckpt_path,
        smpl_npz_path=smpl_path,
        mean_params_path=mean_path,
        device=device,
    )

    # Synthetic test person crop
    img = np.ones((400, 300, 3), dtype=np.uint8) * 128
    box = BoundingBox2D(x_min=0.1, y_min=0.05, x_max=0.9, y_max=0.95)

    result = reconstructor.reconstruct(img, box)

    # 1. Exact SMPL topology specifications
    assert result.vertices.shape == (6890, 3)
    assert result.faces.shape == (13776, 3)
    assert result.normals.shape == (6890, 3)
    assert np.all(np.isfinite(result.vertices))

    # 2. GLB Asset Verification
    assert len(result.glb_bytes) > 200000  # ~300KB
    assert result.glb_bytes[:4] == b"glTF"

    # 3. Reload from GLB bytes
    scene_or_mesh = trimesh.load(io.BytesIO(result.glb_bytes), file_type="glb")
    mesh = list(scene_or_mesh.geometry.values())[0] if hasattr(scene_or_mesh, "geometry") else scene_or_mesh
    assert len(mesh.vertices) == 6890
    assert len(mesh.faces) == 13776

    # 4. Parameters
    assert result.pose_params is not None
    assert len(result.pose_params) == 144
    assert result.shape_params is not None
    assert len(result.shape_params) == 10
    assert "scale" in result.camera_params
