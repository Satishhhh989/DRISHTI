import io
from pathlib import Path
import pytest
from PIL import Image
from app.assets.storage import LocalAssetStore
from app.core.errors import StorageSecurityError, NotFoundError


def test_asset_storage_and_dimension_extraction(tmp_path: Path):
    store = LocalAssetStore(root=tmp_path)

    # Create dummy 200x150 test image
    img = Image.new("RGB", (200, 150), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    content = buf.getvalue()

    meta = store.store_file(
        content=content,
        category="original_photo",
        original_filename="blue_test.jpg",
        mime_type="image/jpeg",
    )

    assert meta.dimensions is not None
    assert meta.dimensions.width == 200
    assert meta.dimensions.height == 150
    assert meta.size_bytes == len(content)
    assert len(meta.checksum_sha256) == 64

    # Verify file can be read back
    path = store.get_file_path(meta.asset_id)
    assert path.exists()
    assert path.read_bytes() == content


def test_path_traversal_prevention(tmp_path: Path):
    store = LocalAssetStore(root=tmp_path)

    # Relative path escaping root must raise StorageSecurityError
    with pytest.raises(StorageSecurityError):
        store._resolve_safe_path(Path("../../etc/passwd"))


def test_non_existent_asset(tmp_path: Path):
    store = LocalAssetStore(root=tmp_path)
    with pytest.raises(NotFoundError):
        store.get_metadata("non-existent-id")
