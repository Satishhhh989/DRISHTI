"""DRISHTI - Skeleton Schema & Hierarchy Validation Tests (Phase 07)
"""

import numpy as np
import pytest

from app.pipeline.motion import (
    CANONICAL_JOINT_NAMES,
    CANONICAL_PARENTS,
    CANONICAL_REST_POSITIONS,
    validate_skeleton_hierarchy,
)
from app.schemas.scene import (
    BoneDefinition,
    CoordinateConvention,
    SkeletonData,
    SkeletonJoint,
)


def create_valid_test_skeleton(entity_id: str = "person-001") -> SkeletonData:
    """Helper to build a valid canonical 24-joint skeleton."""
    joints = []
    bones = []
    for idx, name in enumerate(CANONICAL_JOINT_NAMES):
        parent = CANONICAL_PARENTS[name]
        pos = CANONICAL_REST_POSITIONS[name]
        joints.append(
            SkeletonJoint(
                id=name,
                name=name.replace("_", " ").title(),
                index=idx,
                parent_id=parent,
                position=pos,
                rotation=[0.0, 0.0, 0.0, 1.0],
                confidence=0.92,
                is_observed=True,
            )
        )
        if parent:
            bones.append(
                BoneDefinition(
                    id=f"bone-{parent}-{name}",
                    head_joint_id=parent,
                    tail_joint_id=name,
                    length=0.25,
                )
            )

    return SkeletonData(
        id=f"skel-{entity_id}",
        entity_id=entity_id,
        schema_version="1.0.0",
        root_joint_id="pelvis",
        joints=joints,
        bones=bones,
        coordinate_convention=CoordinateConvention(up="+Y", forward="+Z", right="+X", units="meters"),
        source="smpl_regressed",
        is_aligned=True,
        validation_status="valid",
    )


def test_valid_skeleton_hierarchy():
    """Verifies that a canonically structured 24-joint skeleton passes validation."""
    skel = create_valid_test_skeleton()
    is_valid, issues = validate_skeleton_hierarchy(skel)
    assert is_valid, f"Expected valid skeleton, got issues: {issues}"
    assert len(issues) == 0
    assert len(skel.joints) == 24
    assert len(skel.bones) == 23


def test_skeleton_detects_duplicate_joints():
    """Verifies that duplicate joint IDs are detected and rejected."""
    skel = create_valid_test_skeleton()
    # Add duplicate head joint
    dup_joint = SkeletonJoint(
        id="head",
        name="Head Duplicate",
        index=24,
        parent_id="neck",
        position=[0.0, 0.7, 0.0],
    )
    skel.joints.append(dup_joint)

    is_valid, issues = validate_skeleton_hierarchy(skel)
    assert not is_valid
    assert any("Duplicate joint identifier" in msg for msg in issues)


def test_skeleton_detects_missing_parent():
    """Verifies that invalid or dangling parent references are detected."""
    skel = create_valid_test_skeleton()
    # Point left knee to non-existent parent
    for j in skel.joints:
        if j.id == "left_knee":
            j.parent_id = "ghost_joint_404"

    is_valid, issues = validate_skeleton_hierarchy(skel)
    assert not is_valid
    assert any("references non-existent parent 'ghost_joint_404'" in msg for msg in issues)


def test_skeleton_detects_circular_parent_relationship():
    """Verifies that cycle detection catches circular parent-child loops."""
    skel = create_valid_test_skeleton()
    # Create cycle: spine1 -> pelvis, but make pelvis -> spine1
    for j in skel.joints:
        if j.id == "pelvis":
            j.parent_id = "spine1"

    is_valid, issues = validate_skeleton_hierarchy(skel)
    assert not is_valid
    assert any("Circular parent-child relationship" in msg or "No root joint found" in msg for msg in issues)


def test_skeleton_detects_non_finite_coordinates():
    """Verifies that NaNs and infinite values in joint positions are detected."""
    skel = create_valid_test_skeleton()
    for j in skel.joints:
        if j.id == "left_wrist":
            j.position = [float("nan"), 0.0, float("inf")]

    is_valid, issues = validate_skeleton_hierarchy(skel)
    assert not is_valid
    assert any("Non-finite position coordinates in joint 'left_wrist'" in msg for msg in issues)


def test_skeleton_serialization_and_deserialization():
    """Verifies round-trip Pydantic JSON serialization and model re-validation."""
    skel = create_valid_test_skeleton("person-002")
    raw_dict = skel.model_dump(mode="json")
    reloaded = SkeletonData.model_validate(raw_dict)

    assert reloaded.id == skel.id
    assert reloaded.entity_id == "person-002"
    assert len(reloaded.joints) == 24
    assert reloaded.root_joint_id == "pelvis"
    assert reloaded.joints[0].id == "pelvis"
    assert reloaded.joints[0].parent_id is None
