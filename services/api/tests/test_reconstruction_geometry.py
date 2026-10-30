"""DRISHTI - Human 3D Reconstruction Geometry & Unit Tests (Phase 05)
"""

import io
import numpy as np
import pytest
import torch
import trimesh

from app.pipeline.reconstructor import (
    MockHumanReconstructor,
    compute_mesh_normals,
    export_mesh_glb,
    prepare_person_crop,
    rot6d_to_rotmat,
    validate_mesh_geometry,
)
from app.schemas.common import BoundingBox2D


def test_prepare_person_crop_aspect_preservation():
    """Verifies that rectangular bounding boxes are square-letterboxed without non-uniform stretching."""
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[100:500, 200:400] = 255  # 2:1 vertical rectangle

    box = BoundingBox2D(x_min=0.25, y_min=0.166, x_max=0.5, y_max=0.833)
    crop, meta = prepare_person_crop(img, box, padding=0.2, target_size=(224, 224))

    assert crop.shape == (224, 224, 3)
    assert meta["target_size"] == (224, 224) if "target_size" in meta else crop.shape[:2] == (224, 224)
    assert meta["aspect_ratio"] > 0
    assert not meta["is_clipped"]


def test_prepare_person_crop_boundary_clipping_flag():
    """Verifies that person boxes touching or exceeding boundaries are safely clamped and flagged."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    box = BoundingBox2D(x_min=0.0, y_min=0.0, x_max=0.4, y_max=0.5)
    crop, meta = prepare_person_crop(img, box, padding=0.2, target_size=(224, 224))

    assert crop.shape == (224, 224, 3)
    assert meta["is_clipped"] is True


def test_validate_mesh_geometry_valid_mesh():
    """Verifies that a mathematically sound mesh passes validation with zero fatal errors."""
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0]
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [1, 2, 3],
        [0, 2, 3]
    ], dtype=np.int32)

    is_valid, warnings = validate_mesh_geometry(verts, faces)
    assert is_valid is True


def test_validate_mesh_geometry_rejects_nans_and_infs():
    """Verifies that non-finite coordinates are strictly rejected."""
    verts = np.array([
        [0.0, 0.0, 0.0],
        [np.nan, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    is_valid, warnings = validate_mesh_geometry(verts, faces)
    assert is_valid is False
    assert any("NaN" in w for w in warnings)


def test_validate_mesh_geometry_rejects_out_of_bounds_face_indices():
    """Verifies that out-of-range face indices are strictly caught."""
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ], dtype=np.float32)
    faces = np.array([[0, 1, 99]], dtype=np.int32)

    is_valid, warnings = validate_mesh_geometry(verts, faces)
    assert is_valid is False
    assert any("out of range" in w for w in warnings)


def test_rot6d_to_rotmat_orthogonality():
    """Verifies that Gram-Schmidt orthogonalization produces valid SO(3) rotation matrices."""
    dummy_6d = torch.randn(10, 6)
    R = rot6d_to_rotmat(dummy_6d)

    assert R.shape == (10, 3, 3)
    # Check R * R^T = I
    RT = R.transpose(-1, -2)
    ident = torch.bmm(R, RT)
    I = torch.eye(3).unsqueeze(0).expand(10, -1, -1)
    assert torch.allclose(ident, I, atol=1e-5)

    # Check determinant == +1
    dets = torch.linalg.det(R)
    ones = torch.ones(10)
    assert torch.allclose(dets, ones, atol=1e-5)


def test_glb_export_and_reload():
    """Verifies that export_mesh_glb creates a standard binary GLTF asset loadable by trimesh."""
    cube = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    verts = np.array(cube.vertices, dtype=np.float32)
    faces = np.array(cube.faces, dtype=np.int32)

    glb_bytes = export_mesh_glb(verts, faces, name="test_box")
    assert len(glb_bytes) > 0
    assert glb_bytes[:4] == b"glTF"  # GLB magic header

    # Reload using trimesh
    reloaded = trimesh.load(io.BytesIO(glb_bytes), file_type="glb")
    # For scenes or meshes:
    reloaded_mesh = list(reloaded.geometry.values())[0] if hasattr(reloaded, "geometry") else reloaded
    assert len(reloaded_mesh.vertices) == len(verts)
    assert len(reloaded_mesh.faces) == len(faces)


def test_mock_human_reconstructor_geometry_specs():
    """Verifies that MockHumanReconstructor produces over 1,000 vertices and valid topology."""
    rec = MockHumanReconstructor(device="cpu")
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    box = BoundingBox2D(x_min=0.2, y_min=0.1, x_max=0.8, y_max=0.9)

    result = rec.reconstruct(img, box)
    assert result.vertices.shape[0] >= 1000
    assert result.faces.shape[0] > 1500
    assert result.normals.shape == result.vertices.shape
    assert len(result.glb_bytes) > 1000
    assert result.dimensions[1] > 1.0  # Height > 1.0 meter
    assert result.quality_score > 0.8

