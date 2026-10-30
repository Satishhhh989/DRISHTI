# Development & Testing Guide

## 1. Prerequisites

- **Node.js**: `v20+` (tested on Node `v26.8.2`)
- **Package Manager**: `pnpm` (tested on `11.3.0`)
- **Python**: `3.11+` (tested on `3.11.17`)
- **Python Tooling**: `uv` recommended for environment management

---

## 2. Quick Start

### Initial Setup
Run the setup script to create the Python virtual environment, install backend and frontend dependencies, compile shared packages, and initialize storage:

```bash
bash scripts/setup.sh
```

### Launch Development Environment
To launch both the backend API and frontend workspace concurrently:

```bash
bash scripts/dev.sh
```

- **Frontend Workspace**: `http://localhost:3000`
- **Backend API**: `http://127.0.0.1:8000`
- **Interactive OpenAPI Docs**: `http://127.0.0.1:8000/docs`

Alternatively, run each service independently:
```bash
# Terminal 1: Backend
bash scripts/run_api.sh

# Terminal 2: Frontend
pnpm dev:web
```

---

## 3. Automated Testing

To run the complete test suite across all packages, services, and browsers:

```bash
bash scripts/test.sh
```

This runs:
1. `packages/scene-schema`: Vitest schema validation tests.
2. `packages/shared-types`: Vitest event contract and DTO tests.
3. `apps/web`: Vitest component and store tests.
4. `services/api`: Pytest backend test suite.
5. `tests/e2e`: Playwright browser-level tests.

### Running Individual Test Suites

```bash
# Schema package tests
pnpm --filter @photo-to-3d/scene-schema test

# Shared types tests
pnpm --filter @photo-to-3d/shared-types test

# Frontend unit and component tests
pnpm --filter @photo-to-3d/web test

# Backend Python test suite
bash scripts/test_backend.sh

# Playwright browser tests
pnpm test:e2e
```

---

## 4. Environment Variables

Configuration is driven by `.env` (derived from `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `APP_ENV` | `development` | Application runtime environment |
| `API_HOST` | `127.0.0.1` | Loopback IP address for local privacy |
| `API_PORT` | `8000` | Local port for FastAPI service |
| `API_LOG_LEVEL` | `info` | Structured logging verbosity |
| `STORAGE_ROOT` | `./data/storage` | Local directory for storing photographs and assets |
| `MAX_UPLOAD_SIZE_MB`| `50` | Maximum file upload limit in megabytes |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Allowed origins for web client |
| `NEXT_PUBLIC_API_URL`| `http://127.0.0.1:8000` | Backend API URL accessed by web client |
