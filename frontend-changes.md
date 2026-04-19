# Frontend Changes

No frontend changes were made in this task.

This task enhanced the **backend testing framework** only:

- `backend/tests/conftest.py` — added `mock_rag_system` and `api_client` fixtures; sys.path now includes both `backend/` and `backend/tests/`
- `backend/tests/_helpers.py` — new helper module with `build_test_app()` factory that creates a minimal FastAPI app (same endpoints as `app.py`, no static file mount) wired to a provided RAG instance
- `backend/tests/test_api_endpoints.py` — 17 new tests covering `POST /api/query`, `GET /api/courses`, and `DELETE /api/session/{session_id}` (happy path, error cases, and validation)
- `pyproject.toml` — added `httpx>=0.27.0` dev dependency and `[tool.pytest.ini_options]` with `testpaths`, `pythonpath`, and `-v` flag
