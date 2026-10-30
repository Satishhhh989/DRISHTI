"""DRISHTI - Deterministic Mock Detector & Segmentor

Provides predictable, zero-download mock inference adapters for unit tests,
integration tests, and testing zero-person / partial-failure scenarios.
"""

import io
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image, ImageDraw

from app.pipeline.interfaces import (
    DetectionResult,
    IPersonDetector,
    ISegmentor,
    SegmentationResult,
)
from app.schemas.common import BoundingBox2D


class MockPersonDetector(IPersonDetector):
    """Deterministic mock detector producing fixed test detections."""

    def __init__(self, simulated_people_count: int = 2):
        self.simulated_people_count = simulated_people_count

    def detect_people(
        self,
        image_path: Path,
        confidence_threshold: float = 0.35,
    ) -> List[DetectionResult]:
        with Image.open(image_path) as img:
            w, h = img.size

        if self.simulated_people_count == 0:
            return []

        results: List[DetectionResult] = []
        # Person 1 (left side)
        if self.simulated_people_count >= 1:
            p1_box = (int(w * 0.1), int(h * 0.2), int(w * 0.45), int(h * 0.85))
            results.append(
                DetectionResult(
                    bounding_box=BoundingBox2D(x_min=0.1, y_min=0.2, x_max=0.45, y_max=0.85),
                    pixel_box=p1_box,
                    confidence=0.92,
                    entity_id="mock-person-001",
                    label="person",
                    model="mock-detector-v1",
                )
            )

        # Person 2 (right side)
        if self.simulated_people_count >= 2:
            p2_box = (int(w * 0.55), int(h * 0.25), int(w * 0.9), int(h * 0.9))
            results.append(
                DetectionResult(
                    bounding_box=BoundingBox2D(x_min=0.55, y_min=0.25, x_max=0.9, y_max=0.9),
                    pixel_box=p2_box,
                    confidence=0.87,
                    entity_id="mock-person-002",
                    label="person",
                    model="mock-detector-v1",
                )
            )

        return results


class MockPersonSegmentor(ISegmentor):
    """Deterministic mock segmentor drawing synthetic ellipse masks."""

    def __init__(self, simulate_partial_failure: bool = False):
        self.simulate_partial_failure = simulate_partial_failure

    def segment_people(
        self,
        image_path: Path,
        detections: List[DetectionResult],
    ) -> List[SegmentationResult]:
        with Image.open(image_path) as img:
            w, h = img.size

        results: List[SegmentationResult] = []
        for idx, det in enumerate(detections):
            # If partial failure simulated, skip the second person
            if self.simulate_partial_failure and idx == 1:
                continue

            x1, y1, x2, y2 = det.pixel_box
            # Draw a solid binary mask matching the bounding box
            mask_img = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask_img)
            draw.ellipse([x1, y1, x2, y2], fill=255)

            buf = io.BytesIO()
            mask_img.save(buf, format="PNG")
            mask_bytes = buf.getvalue()

            results.append(
                SegmentationResult(
                    entity_id=det.entity_id,
                    mask_png_bytes=mask_bytes,
                    confidence=det.confidence,
                    pixel_count=1000,
                    width=w,
                    height=h,
                    model="mock-segmentor-v1",
                )
            )

        return results


class MockDepthEstimator:
    """Deterministic mock depth estimator producing smooth synthetic depth maps."""

    def __init__(self, simulate_failure: bool = False):
        self.simulate_failure = simulate_failure

    def estimate_depth(self, image_path: Path):
        if self.simulate_failure:
            raise RuntimeError("Simulated depth estimation failure")

        with Image.open(image_path) as img:
            w, h = img.size

        from app.pipeline.depth import generate_depth_visualization, normalize_depth_map
        from app.pipeline.interfaces import DepthEstimationResult

        # Create smooth synthetic gradient with foreground bump
        y_coords = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        x_coords = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
        base = y_coords * 0.6 + 0.1
        cx, cy = 0.5, 0.55
        bump = np.exp(-(((x_coords - cx) ** 2) / 0.08 + ((y_coords - cy) ** 2) / 0.12)) * 0.35
        raw_depth = (base + bump).astype(np.float32)

        norm, norm_meta = normalize_depth_map(
            raw_depth,
            model_convention="disparity_larger_is_closer",
        )
        vis_bytes = generate_depth_visualization(norm)

        return DepthEstimationResult(
            raw_depth=raw_depth,
            normalized_depth=norm,
            visualization_png_bytes=vis_bytes,
            width=w,
            height=h,
            model_name="mock-depth-v1",
            device="cpu",
            normalization_metadata=norm_meta,
            convention="relative_proximity",
        )
