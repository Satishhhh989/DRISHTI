import io
import pytest
from httpx import AsyncClient, ASGITransport
from PIL import Image
from app.assets.manager import asset_manager
from app.main import app
from app.scene.manager import scene_manager


@pytest.mark.asyncio
async def test_hardware_endpoint():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        res = await client.get("/api/v1/pipeline/hardware")
        assert res.status_code == 200
        data = res.json()
        assert "device" in data
        assert "device_name" in data
        assert "is_accelerated" in data


@pytest.mark.asyncio
async def test_models_endpoint():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        res = await client.get("/api/v1/pipeline/models")
        assert res.status_code == 200
        data = res.json()
        assert "models" in data
        assert "yolov8n-seg" in data["models"]


@pytest.mark.asyncio
async def test_analyze_endpoint_with_mock():
    # Create test scene
    img = Image.new("RGB", (400, 300), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    photo = asset_manager.register_photo(buf.getvalue(), "api_test.jpg")
    scene = scene_manager.create_scene_from_photo(photo)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        res = await client.post(
            "/api/v1/pipeline/analyze",
            json={
                "scene_id": scene.id,
                "confidence_threshold": 0.35,
                "use_mock": True,
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert "job_id" in data
        assert data["scene_id"] == scene.id
        assert data["status"] == "running"

        job_id = data["job_id"]
        # Query job status
        job_res = await client.get(f"/api/v1/jobs/{job_id}")
        assert job_res.status_code == 200
        job_data = job_res.json()
        assert job_data["job_id"] == job_id
