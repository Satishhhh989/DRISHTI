"""DRISHTI - Depth Estimation & Normalization Unit Tests
Tests depth normalization, canonical proximity conversion, pathological input handling,
per-person depth statistics, and visual asset generation.
"""

import io
import numpy as np
from PIL import Image, ImageDraw
import pytest

from app.pipeline.depth import (
    compute_entity_depth_stats,
    generate_depth_visualization,
    normalize_depth_map,
)
from app.schemas.scene import (
    EntityDepthInfo,
    Scene,
    SceneDepthMetadata,
    SpatialPhotoConfig,
)


def test_normalize_standard_relative_distance():
    """Test standard distance-like relative depth (smaller is closer)."""
    # 1.0 is near, 5.0 is far
    raw = np.array([
        [1.0, 2.0],
        [3.0, 5.0]
    ], dtype=np.float32)

    norm, meta = normalize_depth_map(raw, model_convention="distance_smaller_is_closer")

    assert norm.shape == (2, 2)
    assert meta["canonical_convention"] == "relative_proximity"
    assert meta["model_convention"] == "distance_smaller_is_closer"
    # Canonical proximity: higher value must be closer (smaller raw distance)
    assert norm[0, 0] > norm[1, 1], "Raw 1.0 (closer) must have higher proximity than 5.0 (farther)"
    assert 0.0 <= norm.min() <= norm.max() <= 1.0


def test_normalize_disparity_larger_is_closer():
    """Test disparity-like depth (larger is closer)."""
    raw = np.array([
        [10.0, 5.0],
        [2.0, 1.0]
    ], dtype=np.float32)

    norm, meta = normalize_depth_map(raw, model_convention="disparity_larger_is_closer")

    # Raw 10.0 is nearest -> highest proximity
    assert norm[0, 0] > norm[1, 1]
    assert 0.0 <= norm.min() <= norm.max() <= 1.0


def test_normalize_nan_and_infinity_handling():
    """Ensure NaNs and +/- Infinities do not crash or corrupt normalization."""
    raw = np.array([
        [np.nan, 2.0],
        [np.inf, -np.inf],
        [1.0, 4.0]
    ], dtype=np.float32)

    norm, meta = normalize_depth_map(raw, model_convention="distance_smaller_is_closer")

    assert not np.isnan(norm).any(), "Normalized depth must not contain NaNs"
    assert not np.isinf(norm).any(), "Normalized depth must not contain infinities"
    assert 0.0 <= norm.min() <= norm.max() <= 1.0


def test_normalize_constant_depth_field():
    """Test that a uniform/flat depth map produces neutral 0.5 proximity without division by zero."""
    raw = np.full((100, 100), 3.1415, dtype=np.float32)

    norm, meta = normalize_depth_map(raw)

    assert not np.isnan(norm).any()
    assert np.allclose(norm, 0.5), "Flat depth field should produce neutral 0.5 proximity"


def test_normalize_empty_depth_map_raises():
    """Test that empty array raises ValueError."""
    empty = np.array([], dtype=np.float32)
    with pytest.raises(ValueError, match="Cannot normalize empty depth map"):
        normalize_depth_map(empty)


def test_generate_depth_visualization_png():
    """Verify inspection visualization produces a valid 8-bit grayscale PNG."""
    norm = np.linspace(0.0, 1.0, 10000, dtype=np.float32).reshape((100, 100))
    png_bytes = generate_depth_visualization(norm)

    assert len(png_bytes) > 0
    with Image.open(io.BytesIO(png_bytes)) as img:
        assert img.format == "PNG"
        assert img.mode == "L"
        assert img.size == (100, 100)
        # Check that top-left (0.0) is dark and bottom-right (1.0) is bright
        arr = np.array(img)
        assert arr[0, 0] == 0
        assert arr[99, 99] == 255


def test_compute_person_depth_stats_with_mask():
    """Verify calculation of per-person depth statistics using instance mask."""
    # Synthetic depth: left half is near (0.8), right half is far (0.2)
    norm = np.zeros((100, 100), dtype=np.float32)
    norm[:, :50] = 0.8
    norm[:, 50:] = 0.2

    # Person 1 is on the left
    mask_img = Image.new("L", (100, 100), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.rectangle([10, 10, 40, 80], fill=255)
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    mask_bytes = buf.getvalue()

    stats = compute_entity_depth_stats(
        normalized_proximity=norm,
        mask_bytes=mask_bytes,
        pixel_box=[10, 10, 40, 80],
        entity_id="person-001",
    )

    assert stats.status == "ready"
    assert stats.method == "segmentation_mask"
    assert np.isclose(stats.median_proximity, 0.8, atol=0.05)
    assert stats.valid_pixel_coverage > 0.0


def test_compute_person_depth_stats_bounding_box_fallback():
    """Verify fallback to bounding box when mask is absent or empty."""
    norm = np.full((100, 100), 0.65, dtype=np.float32)

    stats = compute_entity_depth_stats(
        normalized_proximity=norm,
        mask_bytes=None,
        pixel_box=[20, 20, 60, 60],
        entity_id="person-fallback",
    )

    assert stats.status == "ready"
    assert stats.method == "bounding_box"
    assert np.isclose(stats.median_proximity, 0.65, atol=0.01)


def test_compute_person_depth_stats_empty_fails_gracefully():
    """Verify graceful handling when neither mask nor bounding box is provided."""
    norm = np.full((50, 50), 0.5, dtype=np.float32)

    stats = compute_entity_depth_stats(
        normalized_proximity=norm,
        mask_bytes=None,
        pixel_box=None,
        entity_id="person-none",
    )

    assert stats.status == "failed"
    assert stats.error is not None

