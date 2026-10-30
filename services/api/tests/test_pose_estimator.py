"""DRISHTI - Pose Estimation Tests (Phase 07)
"""

from pathlib import Path
import numpy as np
import pytest

from app.pipeline.models import model_manager
from app.pipeline.motion import (
    COCO_TO_CANONICAL,
    MockPoseEstimator,
    YoloPoseEstimator,
)
from app.schemas.common import BoundingBox2D


def test_mock_pose_estimator_bounds_and_confidence():
    """Verifies that MockPoseEstimator produces valid normalized keypoints with confidence scores."""
    estimator = MockPoseEstimator()
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    box = BoundingBox2D(x_min=0.2, y_min=0.1, x_max=0.6, y_max=0.85)

    poses = estimator.estimate_poses(dummy_img, [box])
    assert len(poses) == 1
    pose = poses[0]

    assert pose.confidence_score > 0.8
    assert "head" in pose.keypoints
    assert "left_shoulder" in pose.keypoints
    assert "right_ankle" in pose.keypoints

    for name, kp in pose.keypoints.items():
        assert 0.0 <= kp.x_norm <= 1.0, f"Keypoint {name} x out of range: {kp.x_norm}"
        assert 0.0 <= kp.y_norm <= 1.0, f"Keypoint {name} y out of range: {kp.y_norm}"
        assert 0.0 <= kp.confidence <= 1.0, f"Keypoint {name} conf out of range: {kp.confidence}"
        assert kp.is_visible is True


def test_mock_pose_estimator_multiple_people():
    """Verifies multi-person pose estimation maintains separate poses for distinct bounding boxes."""
    estimator = MockPoseEstimator()
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    box_left = BoundingBox2D(x_min=0.05, y_min=0.1, x_max=0.45, y_max=0.9)
    box_right = BoundingBox2D(x_min=0.55, y_min=0.1, x_max=0.95, y_max=0.9)

    poses = estimator.estimate_poses(dummy_img, [box_left, box_right])
    assert len(poses) == 2

    # Left person head should be to the left of right person head
    assert poses[0].keypoints["head"].x_norm < poses[1].keypoints["head"].x_norm


def test_real_yolo_pose_smoke():
    """Runs genuine local YOLOv8-pose inference if model weights exist."""
    pose_path = model_manager.get_model_path("yolov8n-pose")
    if not pose_path.exists():
        pytest.skip("YOLOv8-pose weights not present in data/models/yolov8n-pose.pt; skipping real inference test.")

    estimator = YoloPoseEstimator(model_path=pose_path, device="cpu")
    # Clean blank canvas to test model inference execution
    test_img = np.zeros((320, 320, 3), dtype=np.uint8)
    poses = estimator.estimate_poses(test_img)
    # Blank canvas contains zero people, should return empty list gracefully without throwing
    assert isinstance(poses, list)
