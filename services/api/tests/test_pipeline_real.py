import io
import pytest
from PIL import Image, ImageDraw
from app.assets.manager import asset_manager
from app.pipeline.detector import YoloPersonDetector
from app.pipeline.device import device_manager
from app.pipeline.models import model_manager
from app.pipeline.segmentor import YoloPersonSegmentor
from app.scene.manager import scene_manager


def test_real_model_smoke():
    """Real-model smoke test verifying YOLOv8n-seg detector and segmentor inference."""
    if not model_manager.is_model_available("yolov8n-seg"):
        pytest.skip("Model weights for 'yolov8n-seg' are not locally available. Skipping real-model test.")

    # 1. Detector initialization
    detector = YoloPersonDetector("yolov8n-seg")
    assert detector is not None

    # 2. Selected inference device
    assert device_manager.device in ("mps", "cuda", "cpu")
    assert device_manager.device_name != "Unknown"

    # 3. Create a test image with a simple human-like silhouette to test detector execution
    test_img = Image.new("RGB", (640, 640), color=(230, 230, 230))
    draw = ImageDraw.Draw(test_img)
    # Head and body
    draw.ellipse([270, 100, 370, 200], fill=(50, 50, 50))
    draw.rectangle([250, 200, 390, 500], fill=(30, 30, 30))
    # Legs
    draw.rectangle([260, 500, 310, 620], fill=(20, 20, 20))
    draw.rectangle([330, 500, 380, 620], fill=(20, 20, 20))

    buf = io.BytesIO()
    test_img.save(buf, format="JPEG")
    photo_asset = asset_manager.register_photo(
        content=buf.getvalue(),
        filename="real_smoke_test.jpg",
    )
    photo_path = asset_manager.get_asset_path(photo_asset.asset_id)

    # 4. Detector can process the image
    detections = detector.detect_people(photo_path, confidence_threshold=0.1)
    assert isinstance(detections, list)

    # 5. Segmentor initialization
    segmentor = YoloPersonSegmentor("yolov8n-seg")
    assert segmentor is not None

    # 6. Run segmentation if any detection was produced
    if detections:
        seg_results = segmentor.segment_people(photo_path, detections)
        assert isinstance(seg_results, list)
        if seg_results:
            first_mask = seg_results[0]
            assert first_mask.width == 640
            assert first_mask.height == 640
            assert len(first_mask.mask_png_bytes) > 0
