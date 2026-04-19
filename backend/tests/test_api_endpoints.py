"""
Tests for the FastAPI endpoints: POST /api/query, GET /api/courses,
DELETE /api/session/{session_id}.

Uses a minimal test app built from conftest.build_test_app() so that the
real app.py (which mounts static files from ../frontend) is never imported.
"""
import pytest
from unittest.mock import MagicMock
from _helpers import build_test_app
from fastapi.testclient import TestClient


# ── POST /api/query ────────────────────────────────────────────────────────

class TestQueryEndpoint:
    def test_returns_200_with_valid_query(self, api_client):
        resp = api_client.post("/api/query", json={"query": "What is MCP?"})
        assert resp.status_code == 200

    def test_response_contains_answer(self, api_client):
        resp = api_client.post("/api/query", json={"query": "What is MCP?"})
        assert resp.json()["answer"] == "Test answer"

    def test_response_contains_sources(self, api_client):
        resp = api_client.post("/api/query", json={"query": "What is MCP?"})
        sources = resp.json()["sources"]
        assert isinstance(sources, list)
        assert sources[0]["label"] == "Course A - Lesson 1"

    def test_response_contains_session_id(self, api_client):
        resp = api_client.post("/api/query", json={"query": "What is MCP?"})
        assert resp.json()["session_id"] == "test-session-id"

    def test_provided_session_id_is_forwarded(self, api_client, mock_rag_system):
        api_client.post("/api/query", json={"query": "hi", "session_id": "my-sid"})
        mock_rag_system.query.assert_called_once_with("hi", "my-sid")

    def test_missing_session_id_creates_new_session(self, api_client, mock_rag_system):
        api_client.post("/api/query", json={"query": "hi"})
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_missing_query_field_returns_422(self, api_client):
        resp = api_client.post("/api/query", json={})
        assert resp.status_code == 422

    def test_rag_exception_returns_500(self, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("db failure")
        app = build_test_app(mock_rag_system)
        with TestClient(app) as client:
            resp = client.post("/api/query", json={"query": "boom", "session_id": "s"})
        assert resp.status_code == 500
        assert "db failure" in resp.json()["detail"]

    def test_empty_sources_list_is_valid(self, mock_rag_system):
        mock_rag_system.query.return_value = ("answer", [])
        app = build_test_app(mock_rag_system)
        with TestClient(app) as client:
            resp = client.post("/api/query", json={"query": "q", "session_id": "s"})
        assert resp.status_code == 200
        assert resp.json()["sources"] == []


# ── GET /api/courses ───────────────────────────────────────────────────────

class TestCoursesEndpoint:
    def test_returns_200(self, api_client):
        resp = api_client.get("/api/courses")
        assert resp.status_code == 200

    def test_total_courses_count(self, api_client):
        assert api_client.get("/api/courses").json()["total_courses"] == 2

    def test_course_titles_list(self, api_client):
        titles = api_client.get("/api/courses").json()["course_titles"]
        assert titles == ["Course A", "Course B"]

    def test_analytics_exception_returns_500(self, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("analytics error")
        app = build_test_app(mock_rag_system)
        with TestClient(app) as client:
            resp = client.get("/api/courses")
        assert resp.status_code == 500

    def test_empty_course_list(self, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        app = build_test_app(mock_rag_system)
        with TestClient(app) as client:
            resp = client.get("/api/courses")
        assert resp.status_code == 200
        assert resp.json()["total_courses"] == 0
        assert resp.json()["course_titles"] == []


# ── DELETE /api/session/{session_id} ──────────────────────────────────────

class TestDeleteSessionEndpoint:
    def test_returns_200(self, api_client):
        resp = api_client.delete("/api/session/abc-123")
        assert resp.status_code == 200

    def test_returns_ok_status(self, api_client):
        resp = api_client.delete("/api/session/abc-123")
        assert resp.json() == {"status": "ok"}

    def test_delegates_to_session_manager(self, api_client, mock_rag_system):
        api_client.delete("/api/session/my-session")
        mock_rag_system.session_manager.clear_session.assert_called_once_with("my-session")
