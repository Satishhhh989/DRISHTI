import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.mark.asyncio
async def test_version_endpoint():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/version")
        assert response.status_code == 200
        data = response.json()
        assert data["service_version"] == "0.1.0"
        assert data["api_version"] == "v1"
        assert data["schema_version"] == "1.0.0"

