"""
Tests for RAGSystem content-query handling — covers the full query() pipeline
for the Gemini provider path and provider-agnostic integration tests.
"""
import os
import sys
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from vector_store import SearchResults


# ── config stubs ───────────────────────────────────────────────────────────

class GeminiStubConfig:
    ANTHROPIC_API_KEY = ""
    ANTHROPIC_MODEL = ""
    GOOGLE_API_KEY = "gemini_test_key"
    GEMINI_MODEL = "gemini-test-model"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    MAX_RESULTS = 5
    MAX_HISTORY = 2
    CHROMA_PATH = "/tmp/stub_chroma"


# ── shared fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.search.return_value = SearchResults(
        documents=["Lesson 1 content: MCP protocol explained."],
        metadata=[{"course_title": "MCP Course", "lesson_number": 1}],
        distances=[0.1],
    )
    store.get_lesson_link.return_value = "https://example.com/lesson/1"
    store.get_existing_course_titles.return_value = ["MCP Course"]
    store.get_course_count.return_value = 1
    return store


@pytest.fixture
def mock_generator():
    gen = MagicMock()
    gen.generate_response.return_value = "MCP is a protocol for context sharing."
    return gen


@pytest.fixture
def rag_gemini(mock_vector_store, mock_generator):
    with patch("rag_system.VectorStore", return_value=mock_vector_store), \
         patch("rag_system.GeminiAIGenerator", return_value=mock_generator), \
         patch("rag_system.DocumentProcessor"):
        from rag_system import RAGSystem
        system = RAGSystem(GeminiStubConfig(), provider="gemini")
        system.vector_store = mock_vector_store
        system.ai_generator = mock_generator
        return system


# ── Gemini provider: query() contract ─────────────────────────────────────

def test_gemini_query_returns_tuple_of_str_and_list(rag_gemini):
    answer, sources = rag_gemini.query("What is MCP?")
    assert isinstance(answer, str)
    assert isinstance(sources, list)


def test_gemini_query_answer_is_generator_output(rag_gemini, mock_generator):
    answer, _ = rag_gemini.query("What is MCP?")
    assert answer == "MCP is a protocol for context sharing."


def test_gemini_query_passes_tool_definitions_to_generator(rag_gemini, mock_generator):
    rag_gemini.query("What is MCP?")
    kwargs = mock_generator.generate_response.call_args.kwargs
    assert "tools" in kwargs
    tool_names = [t["name"] for t in kwargs["tools"]]
    assert "search_course_content" in tool_names
    assert "get_course_outline" in tool_names


def test_gemini_query_passes_tool_manager_to_generator(rag_gemini, mock_generator):
    rag_gemini.query("What is MCP?")
    kwargs = mock_generator.generate_response.call_args.kwargs
    assert "tool_manager" in kwargs
    assert kwargs["tool_manager"] is rag_gemini.tool_manager


def test_gemini_registers_both_tools(rag_gemini):
    assert "search_course_content" in rag_gemini.tool_manager.tools
    assert "get_course_outline" in rag_gemini.tool_manager.tools


def test_gemini_uses_gemini_generator_not_anthropic(mock_vector_store, mock_generator):
    """RAGSystem with provider='gemini' must instantiate GeminiAIGenerator, not AIGenerator."""
    with patch("rag_system.VectorStore", return_value=mock_vector_store), \
         patch("rag_system.GeminiAIGenerator", return_value=mock_generator) as MockGemini, \
         patch("rag_system.AIGenerator") as MockAnthropic, \
         patch("rag_system.DocumentProcessor"):
        from rag_system import RAGSystem
        RAGSystem(GeminiStubConfig(), provider="gemini")
        MockGemini.assert_called_once()
        MockAnthropic.assert_not_called()


# ── source lifecycle ───────────────────────────────────────────────────────

def test_gemini_query_returns_sources_from_search_tool(rag_gemini):
    search_tool = rag_gemini.tool_manager.tools["search_course_content"]
    search_tool.last_sources = [{"label": "MCP Course - Lesson 1", "url": "https://example.com"}]
    _, sources = rag_gemini.query("What is MCP?")
    assert sources == [{"label": "MCP Course - Lesson 1", "url": "https://example.com"}]


def test_gemini_query_resets_sources_after_retrieval(rag_gemini):
    search_tool = rag_gemini.tool_manager.tools["search_course_content"]
    search_tool.last_sources = [{"label": "stale", "url": None}]
    rag_gemini.query("What is MCP?")
    assert search_tool.last_sources == []


def test_gemini_query_returns_empty_sources_when_no_search_ran(rag_gemini):
    rag_gemini.tool_manager.tools["search_course_content"].last_sources = []
    _, sources = rag_gemini.query("What is Python?")
    assert sources == []


# ── session history ────────────────────────────────────────────────────────

def test_gemini_query_stores_exchange_in_session(rag_gemini):
    sid = rag_gemini.session_manager.create_session()
    rag_gemini.query("What is MCP?", session_id=sid)
    history = rag_gemini.session_manager.get_conversation_history(sid)
    assert history is not None
    assert "What is MCP?" in history


def test_gemini_query_passes_history_on_second_call(rag_gemini, mock_generator):
    sid = rag_gemini.session_manager.create_session()
    rag_gemini.query("First question", session_id=sid)
    rag_gemini.query("Second question", session_id=sid)
    second_kwargs = mock_generator.generate_response.call_args_list[1].kwargs
    assert second_kwargs.get("conversation_history") is not None


def test_gemini_query_without_session_id_still_works(rag_gemini):
    answer, sources = rag_gemini.query("What is MCP?")
    assert answer is not None


# ── integration: real ChromaDB, mocked generator ──────────────────────────

DOCS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "docs",
)


@pytest.fixture(scope="module")
def integrated_rag():
    """
    Real RAGSystem with a temporary ChromaDB populated from actual docs.
    The AI generator is mocked so no API key is required.
    """
    txt_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".txt")]
    if not txt_files:
        pytest.skip("No .txt files in docs/ — skipping integration tests")

    with tempfile.TemporaryDirectory() as tmp:
        mock_gen = MagicMock()
        mock_gen.generate_response.return_value = "Mocked final answer."

        with patch("rag_system.GeminiAIGenerator", return_value=mock_gen):
            from rag_system import RAGSystem

            cfg = type("C", (), {
                "ANTHROPIC_API_KEY": "",
                "ANTHROPIC_MODEL": "",
                "GOOGLE_API_KEY": "x",
                "GEMINI_MODEL": "gemini-test-model",
                "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
                "CHUNK_SIZE": 800,
                "CHUNK_OVERLAP": 100,
                "MAX_RESULTS": 5,
                "MAX_HISTORY": 2,
                "CHROMA_PATH": tmp,
            })()

            system = RAGSystem(cfg, provider="gemini")
            courses, _ = system.add_course_folder(DOCS_DIR, clear_existing=False)
            assert courses > 0, "No courses loaded — document parsing may be broken"
            system.ai_generator = mock_gen
            yield system


def test_integration_courses_loaded(integrated_rag):
    analytics = integrated_rag.get_course_analytics()
    assert analytics["total_courses"] > 0


def test_integration_search_tool_returns_content(integrated_rag):
    """
    CourseSearchTool must return content, not 'Search error'.
    A Search error means ChromaDB raised (n_results > collection size) and
    the exception propagated — which would surface as HTTP 500 / 'query failed'.
    """
    result = integrated_rag.tool_manager.tools["search_course_content"].execute("introduction")
    assert "Search error" not in result, f"ChromaDB error propagated: {result!r}"


def test_integration_query_does_not_raise(integrated_rag):
    try:
        integrated_rag.query("What topics are covered?")
    except Exception as exc:
        pytest.fail(f"rag_system.query() raised unexpectedly: {exc!r}")


def test_integration_search_with_course_name_filter(integrated_rag):
    first_title = integrated_rag.get_course_analytics()["course_titles"][0]
    result = integrated_rag.tool_manager.tools["search_course_content"].execute(
        "introduction", course_name=first_title[:5]
    )
    assert "Search error" not in result, f"Filtered search failed: {result!r}"


def test_integration_outline_tool_registered(integrated_rag):
    assert "get_course_outline" in integrated_rag.tool_manager.tools


def test_integration_outline_tool_execute(integrated_rag):
    outline_tool = integrated_rag.tool_manager.tools["get_course_outline"]
    first_title = integrated_rag.get_course_analytics()["course_titles"][0]
    result = outline_tool.execute(first_title[:6])
    assert "No course found" not in result, f"Outline tool failed: {result!r}"
    assert "Lesson" in result, f"Outline missing lessons: {result!r}"
