import sys
import os

_tests_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.dirname(_tests_dir)

# Add backend/ and backend/tests/ to path
sys.path.insert(0, _backend_dir)
sys.path.insert(0, _tests_dir)

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from _helpers import build_test_app


# ── shared RAG mock ────────────────────────────────────────────────────────

@pytest.fixture
def mock_rag_system():
    rag = MagicMock()
    rag.query.return_value = ("Test answer", [{"label": "Course A - Lesson 1", "url": "https://example.com/1"}])
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course A", "Course B"],
    }
    rag.session_manager.create_session.return_value = "test-session-id"
    rag.session_manager.clear_session.return_value = None
    return rag


@pytest.fixture
def api_client(mock_rag_system):
    app = build_test_app(mock_rag_system)
    with TestClient(app) as client:
        yield client
