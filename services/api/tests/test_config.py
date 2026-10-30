from app.core.config import Settings


def test_default_config_loopback():
    cfg = Settings(API_HOST="127.0.0.1", API_PORT=8000)
    assert cfg.api_host == "127.0.0.1"
    assert cfg.api_port == 8000
    assert cfg.schema_version == "1.0.0"
    assert cfg.max_upload_size_bytes == 50 * 1024 * 1024


def test_cors_origins_parsing():
    cfg = Settings(CORS_ORIGINS="http://localhost:3000, http://127.0.0.1:3000")
    assert "http://localhost:3000" in cfg.cors_origins
    assert "http://127.0.0.1:3000" in cfg.cors_origins

