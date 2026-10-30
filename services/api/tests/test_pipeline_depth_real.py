"""DRISHTI - Real Model Depth Estimation Smoke Test
Executes real Depth Anything V2 Small neural inference on host hardware (Apple Silicon MPS / CUDA / CPU)
to verify genuine depth outputs, tensor interpolation, normalization, and visual PNG generation.
"""

import io
from pathlib import Path
from PIL import Image
import pytest

from app.pipeline.depth import DepthAnythingV2Estimator
from app.pipeline.models import model_manager


def test_real_depth_model_smoke(tmp_path: Path):
    """Executes genuine local neural network depth inference on a test image."""
    if not model_manager.is_model_available("depth-anything-v2-small"):
        pytest.skip("Depth Anything V2 weights not found locally; skipping real model smoke test.")

    # Create a realistic test image with foreground subject and background
    img = Image.new("RGB", (320, 240), color=(60, 80, 100))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    # Draw a mock person silhouette
    draw.rectangle([100, 40, 220, 220], fill=(220, 180, 140))
    img_path = tmp_path / "real_depth_smoke.jpg"
    img.save(img_path, format="JPEG")

    estimator = DepthAnythingV2Estimator("depth-anything-v2-small")
    result = estimator.estimate_depth(img_path)

    # 1. Dimension alignment
    assert result.width == 320
    assert result.height == 240
    assert result.raw_depth.shape == (240, 320)
    assert result.normalized_depth.shape == (240, 320)

    # 2. Value validation
    assert not np.isnan(result.normalized_depth).any()
    assert 0.0 <= result.normalized_depth.min() <= result.normalized_depth.max() <= 1.0

    # 3. Model convention check
    assert result.convention == "relative_proximity"
    assert result.model_name == "depth-anything-v2-small"
    assert result.device in ("mps", "cuda", "cpu")

    # 4. Visualization PNG validation
    assert len(result.visualization_png_bytes) > 0
    with Image.open(io.BytesIO(result.visualization_png_bytes)) as vis_img:
        assert vis_img.format == "PNG"
        assert vis_img.mode == "L"
        assert vis_img.size == (320, 240)


import numpy as np

