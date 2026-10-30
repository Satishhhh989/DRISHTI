"""DRISHTI - Security Hardening & Integrity Tests
Phase 10: Integration, Testing, Security Hardening, Final Verification & Release
"""

import io
import zipfile
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.assets.validation import validate_image_file, MAX_DIMENSION, MAX_PIXELS
from app.core.errors import ValidationAppError
from app.main import app

client = TestClient(app)


def test_security_headers_present():
    """Verifies that security headers are injected on all HTTP responses."""
    res = client.get("/")
    assert res.status_code == 200
    assert res.headers.get("x-content-type-options") == "nosniff"
    assert res.headers.get("x-frame-options") == "DENY"
    assert "strict-origin" in res.headers.get("referrer-policy", "").lower()
    assert res.headers.get("x-xss-protection") == "1; mode=block"


def test_reject_empty_file_upload():
    """Verifies that 0-byte file uploads are rejected safely."""
    with pytest.raises(ValidationAppError, match="empty"):
        validate_image_file(b"", "empty.jpg")


def test_reject_excessive_upload_file_size():
    """Verifies that uploads exceeding the 50MB limit are rejected."""
    # 51MB dummy payload
    oversized = b"0" * (51 * 1024 * 1024)
    with pytest.raises(ValidationAppError, match="exceeds maximum permitted limit"):
        validate_image_file(oversized, "giant.jpg")


def test_reject_decompression_bomb_dimensions():
    """Verifies that images exceeding maximum dimension (8192px) are rejected."""
    buf = io.BytesIO()
    # Create image with dimension 8193 x 100
    img = Image.new("RGB", (MAX_DIMENSION + 1, 100), color="blue")
    img.save(buf, format="JPEG")
    bytes_data = buf.getvalue()

    with pytest.raises(ValidationAppError, match="exceed maximum allowed dimension"):
        validate_image_file(bytes_data, "wide_bomb.jpg")


def test_reject_animated_images():
    """Verifies that multi-frame animated images are rejected."""
    buf = io.BytesIO()
    frame1 = Image.new("RGBA", (100, 100), color="red")
    frame2 = Image.new("RGBA", (100, 100), color="blue")
    frame1.save(
        buf,
        format="WEBP",
        save_all=True,
        append_images=[frame2],
        duration=200,
        loop=0,
    )
    with pytest.raises(ValidationAppError, match="Animated images are not supported"):
        validate_image_file(buf.getvalue(), "animated.webp")


def test_reject_zip_slip_path_traversal():
    """Verifies that project archives with path traversal are rejected during import."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../../etc/passwd", "malicious_content")
        zf.writestr("project.json", "{}")
        zf.writestr("scene.json", "{}")

    files = {"file": ("exploit.zip", buf.getvalue(), "application/zip")}
    res = client.post("/api/v1/projects/import", files=files)
    assert res.status_code == 422
    assert "path traversal" in res.json()["error"]["message"].lower()


def test_reject_zip_bomb_excessive_entries():
    """Verifies that archives containing more than 250 entries are rejected."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(251):
            zf.writestr(f"file_{i}.txt", "test")
        zf.writestr("project.json", "{}")
        zf.writestr("scene.json", "{}")

    files = {"file": ("entry_bomb.zip", buf.getvalue(), "application/zip")}
    res = client.post("/api/v1/projects/import", files=files)
    assert res.status_code == 422
    assert "too many entries" in res.json()["error"]["message"].lower()


def test_reject_zip_bomb_excessive_uncompressed_size():
    """Verifies that archives with individual entry size > 100MB are rejected."""
    buf = io.BytesIO()
    # Write a zip entry with reported uncompressed size > 100MB
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 105MB of zeros compresses to ~100KB
        zf.writestr("giant_entry.bin", b"\x00" * (105 * 1024 * 1024))
        zf.writestr("project.json", "{}")
        zf.writestr("scene.json", "{}")

    files = {"file": ("size_bomb.zip", buf.getvalue(), "application/zip")}
    res = client.post("/api/v1/projects/import", files=files)
    assert res.status_code == 422
    assert "exceeds maximum allowed size" in res.json()["error"]["message"].lower()


def test_cors_headers_loopback():
    """Verifies that allowed loopback origins receive standard CORS headers."""
    headers = {"Origin": "http://localhost:3000"}
    res = client.options("/api/v1/scenes", headers=headers)
    assert res.headers.get("access-control-allow-origin") == "http://localhost:3000"

