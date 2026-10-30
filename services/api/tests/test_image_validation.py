import io
import pytest
from PIL import Image
from app.assets.validation import validate_image_file, MAX_DIMENSION, MAX_PIXELS
from app.core.errors import ValidationAppError


def test_validate_valid_jpeg():
    img = Image.new("RGB", (640, 480), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    content = buf.getvalue()

    info = validate_image_file(content, "photo.jpg")
    assert info.format == "JPEG"
    assert info.mime_type == "image/jpeg"
    assert info.width == 640
    assert info.height == 480
    assert info.aspect_ratio == round(640 / 480, 6)
    assert info.size_bytes == len(content)


def test_validate_valid_png():
    img = Image.new("RGBA", (300, 300), color=(0, 255, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    content = buf.getvalue()

    info = validate_image_file(content, "transparent.png")
    assert info.format == "PNG"
    assert info.mime_type == "image/png"
    assert info.width == 300
    assert info.height == 300


def test_reject_empty_file():
    with pytest.raises(ValidationAppError) as exc_info:
        validate_image_file(b"", "empty.jpg")
    assert "empty" in str(exc_info.value).lower()


def test_reject_corrupted_file():
    corrupted = b"NOT_A_REAL_IMAGE_FILE_DATA_HERE"
    with pytest.raises(ValidationAppError) as exc_info:
        validate_image_file(corrupted, "corrupted.jpg")
    assert "corrupted or invalid" in str(exc_info.value).lower()


def test_reject_unsupported_format():
    # Create BMP
    img = Image.new("RGB", (100, 100), color=(0, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    content = buf.getvalue()

    with pytest.raises(ValidationAppError) as exc_info:
        validate_image_file(content, "image.bmp")
    assert "unsupported" in str(exc_info.value).lower()


def test_reject_excessive_dimensions():
    # Create an image that exceeds MAX_DIMENSION (8192px)
    img = Image.new("L", (MAX_DIMENSION + 1, 10), color=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    content = buf.getvalue()

    with pytest.raises(ValidationAppError) as exc_info:
        validate_image_file(content, "oversized.png")
    assert "exceed" in str(exc_info.value).lower()
