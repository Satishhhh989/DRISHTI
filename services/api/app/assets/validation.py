"""DRISHTI - Image Validation & Decompression Bomb Protection Engine

Enforces strict file format verification, dimension bounds, corruption checks,
animated-file rejection, and EXIF orientation normalization.
"""

import io
from dataclasses import dataclass
from typing import Optional, Tuple
from PIL import Image, ImageOps

from app.core.config import settings
from app.core.errors import ValidationAppError
from app.core.logging import logger

# Protect against decompression bomb attacks
MAX_PIXELS = 50_000_000  # 50 Megapixels
MAX_DIMENSION = 8192
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


@dataclass
class ValidatedImageInfo:
    format: str
    mime_type: str
    width: int
    height: int
    aspect_ratio: float
    size_bytes: int
    normalized_bytes: Optional[bytes] = None


def validate_image_file(content: bytes, filename: str) -> ValidatedImageInfo:
    """Validates raw image bytes strictly against format, size, dimension, and integrity rules."""
    size_bytes = len(content)

    # 1. Reject empty files
    if size_bytes == 0:
        raise ValidationAppError("Uploaded file is empty (0 bytes).")

    # 2. Enforce upload size limit
    if size_bytes > settings.max_upload_size_bytes:
        raise ValidationAppError(
            f"File size ({size_bytes / (1024 * 1024):.1f} MB) exceeds maximum permitted limit ({settings.max_upload_size_mb} MB)."
        )

    # 3. Verify format and integrity using PIL
    try:
        buf = io.BytesIO(content)
        with Image.open(buf) as test_img:
            # Verify file structure
            test_img.verify()
            detected_format = (test_img.format or "").upper()
    except Exception as e:
        logger.warning(f"Image integrity verification failed for '{filename}': {e}")
        raise ValidationAppError(f"Corrupted or invalid image file: {str(e)}") from e

    # 4. Enforce supported formats
    if detected_format not in ALLOWED_FORMATS:
        raise ValidationAppError(
            f"Unsupported image format: '{detected_format}'. DRISHTI requires JPEG, PNG, or WebP."
        )

    # 5. Full decode inspection (after verify, re-open)
    try:
        buf_decode = io.BytesIO(content)
        with Image.open(buf_decode) as img:
            # Check for multi-frame animations (e.g. animated WebP)
            if getattr(img, "is_animated", False) and getattr(img, "n_frames", 1) > 1:
                raise ValidationAppError(
                    "Animated images are not supported in Phase 02. Please upload a static photograph."
                )

            # Apply EXIF transpose so pixel dimensions match physical orientation
            transposed_img = ImageOps.exif_transpose(img)
            width, height = transposed_img.size

            # Enforce dimension bounds
            if width <= 0 or height <= 0:
                raise ValidationAppError("Image has invalid zero or negative dimensions.")

            if width > MAX_DIMENSION or height > MAX_DIMENSION:
                raise ValidationAppError(
                    f"Image dimensions ({width}x{height}) exceed maximum allowed dimension ({MAX_DIMENSION}px)."
                )

            if (width * height) > MAX_PIXELS:
                raise ValidationAppError(
                    f"Total image resolution ({width * height} pixels) exceeds maximum allowed limit ({MAX_PIXELS} pixels)."
                )

            aspect = round(width / height, 6)

            # If EXIF orientation altered the image or image wasn't RGB/RGBA, prepare normalized bytes
            normalized_bytes: Optional[bytes] = None
            if transposed_img.size != img.size or transposed_img.mode not in ("RGB", "RGBA"):
                save_buf = io.BytesIO()
                # Preserve transparency if RGBA, otherwise RGB JPEG/PNG
                if transposed_img.mode == "RGBA":
                    transposed_img.save(save_buf, format="PNG")
                else:
                    rgb_img = transposed_img.convert("RGB")
                    rgb_img.save(save_buf, format="JPEG", quality=95)
                normalized_bytes = save_buf.getvalue()

            mime_map = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
            mime_type = mime_map.get(detected_format, "image/jpeg")

            return ValidatedImageInfo(
                format=detected_format,
                mime_type=mime_type,
                width=width,
                height=height,
                aspect_ratio=aspect,
                size_bytes=size_bytes,
                normalized_bytes=normalized_bytes,
            )
    except ValidationAppError:
        raise
    except Exception as e:
        logger.warning(f"Image decoding failed for '{filename}': {e}")
        raise ValidationAppError(f"Failed to decode image data: {str(e)}") from e
