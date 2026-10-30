import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.mark.asyncio
async def test_not_found_error_response():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/scenes/non-existent-scene-uuid")
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "RESOURCE_NOT_FOUND"
        assert "message" in data["error"]


@pytest.mark.asyncio
async def test_validation_error_response():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Invalid body to /scenes
        response = await client.post("/api/v1/scenes", json={})
        assert response.status_code == 422

