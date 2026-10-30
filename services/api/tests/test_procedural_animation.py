"""DRISHTI - Procedural Animation Engine Tests (Phase 07)
"""

import math
import numpy as np
import pytest

from app.pipeline.motion import (
    ProceduralAnimationEngine,
    euler_to_quaternion,
    procedural_animation_engine,
)
from app.schemas.scene import AnimationClipData


def test_euler_to_quaternion_unit_norm():
    """Verifies that Euler to quaternion conversion always produces normalized unit quaternions."""
    test_angles = [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.0, -0.3, 0.0),
        (0.2, 0.4, -0.1),
        (math.pi / 2, 0.0, math.pi / 4),
    ]

    for rx, ry, rz in test_angles:
        q = euler_to_quaternion(rx, ry, rz)
        assert len(q) == 4
        norm = np.linalg.norm(q)
        assert np.isclose(norm, 1.0, atol=1e-5), f"Quaternion {q} is not unit norm: {norm}"


def test_procedural_animation_clips_generation():
    """Verifies generation of the 4 standard deterministic humanoid clips."""
    clips = procedural_animation_engine.generate_clips_for_entity("person-001", "skel-person-001")
    assert len(clips) == 4

    clip_names = {c.name for c in clips}
    assert "idle_breathing" in clip_names
    assert "torso_twist" in clip_names
    assert "arm_wave" in clip_names
    assert "gentle_walk" in clip_names

    for clip in clips:
        assert clip.target_entity_id == "person-001"
        assert clip.target_skeleton_id == "skel-person-001"
        assert clip.duration_seconds > 0.0
        assert clip.fps == 30.0
        assert clip.is_looping is True
        assert clip.provenance == "procedural"  # Disclaims recovered motion
        assert len(clip.tracks) > 0

        for track in clip.tracks:
            assert len(track.timestamps) > 0
            assert len(track.rotations) == len(track.timestamps)
            # Timestamps must be strictly ascending
            for i in range(len(track.timestamps) - 1):
                assert track.timestamps[i] < track.timestamps[i + 1]


def test_multiple_people_independent_animations():
    """Verifies that multiple people receive separate clip sets with matching entity IDs."""
    clips_p1 = procedural_animation_engine.generate_clips_for_entity("person-001", "skel-001")
    clips_p2 = procedural_animation_engine.generate_clips_for_entity("person-002", "skel-002")

    assert clips_p1[0].target_entity_id == "person-001"
    assert clips_p2[0].target_entity_id == "person-002"
    assert clips_p1[0].id != clips_p2[0].id

