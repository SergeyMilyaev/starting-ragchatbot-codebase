"""
Tests for CourseSearchTool.execute() — covers happy path, empty results,
error propagation, source tracking, and filter forwarding.
"""
import pytest
from unittest.mock import MagicMock
from search_tools import CourseSearchTool
from vector_store import SearchResults


# ── helpers ────────────────────────────────────────────────────────────────

def make_results(docs, metas, dists=None):
    if dists is None:
        dists = [0.1] * len(docs)
    return SearchResults(documents=docs, metadata=metas, distances=dists)


# ── fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_store():
    store = MagicMock()
    store.get_lesson_link.return_value = "https://example.com/lesson/1"
    return store


@pytest.fixture
def tool(mock_store):
    return CourseSearchTool(mock_store)


# ── happy path ─────────────────────────────────────────────────────────────

def test_execute_returns_course_and_content(tool, mock_store):
    mock_store.search.return_value = make_results(
        ["MCP protocol lets models call external tools."],
        [{"course_title": "MCP Course", "lesson_number": 1}],
    )
    result = tool.execute("MCP protocol")
    assert "MCP Course" in result
    assert "Lesson 1" in result
    assert "MCP protocol lets models" in result


def test_execute_multiple_results_joined(tool, mock_store):
    mock_store.search.return_value = make_results(
        ["First chunk.", "Second chunk."],
        [
            {"course_title": "MCP Course", "lesson_number": 1},
            {"course_title": "MCP Course", "lesson_number": 2},
        ],
    )
    result = tool.execute("query")
    assert "First chunk." in result
    assert "Second chunk." in result


def test_execute_result_without_lesson_number(tool, mock_store):
    """Metadata may omit lesson_number — must not raise."""
    mock_store.search.return_value = make_results(
        ["Content without lesson."],
        [{"course_title": "General Course"}],
    )
    result = tool.execute("query")
    assert "General Course" in result
    assert "Content without lesson." in result


# ── source tracking ────────────────────────────────────────────────────────

def test_execute_stores_sources_with_label_and_url(tool, mock_store):
    mock_store.search.return_value = make_results(
        ["content"],
        [{"course_title": "Test Course", "lesson_number": 3}],
    )
    mock_store.get_lesson_link.return_value = "https://example.com/lesson/3"

    tool.execute("query")

    assert len(tool.last_sources) == 1
    src = tool.last_sources[0]
    assert src["label"] == "Test Course - Lesson 3"
    assert src["url"] == "https://example.com/lesson/3"


def test_execute_no_lesson_number_sets_url_to_none(tool, mock_store):
    mock_store.search.return_value = make_results(
        ["content"],
        [{"course_title": "Test Course"}],  # no lesson_number
    )
    tool.execute("query")
    assert tool.last_sources[0]["url"] is None


def test_execute_overwrites_last_sources_on_each_call(tool, mock_store):
    mock_store.search.return_value = make_results(
        ["content"],
        [{"course_title": "Course A", "lesson_number": 1}],
    )
    tool.execute("first query")
    assert tool.last_sources[0]["label"] == "Course A - Lesson 1"

    mock_store.search.return_value = make_results(
        ["content"],
        [{"course_title": "Course B", "lesson_number": 2}],
    )
    tool.execute("second query")
    assert len(tool.last_sources) == 1
    assert tool.last_sources[0]["label"] == "Course B - Lesson 2"


# ── filter forwarding ──────────────────────────────────────────────────────

def test_execute_forwards_course_name_to_store(tool, mock_store):
    mock_store.search.return_value = make_results([], [])
    tool.execute("MCP", course_name="MCP Course")
    mock_store.search.assert_called_once_with(
        query="MCP", course_name="MCP Course", lesson_number=None
    )


def test_execute_forwards_lesson_number_to_store(tool, mock_store):
    mock_store.search.return_value = make_results([], [])
    tool.execute("MCP", lesson_number=5)
    mock_store.search.assert_called_once_with(
        query="MCP", course_name=None, lesson_number=5
    )


# ── empty results ──────────────────────────────────────────────────────────

def test_execute_empty_results_returns_not_found(tool, mock_store):
    mock_store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
    result = tool.execute("nothing here")
    assert "No relevant content found" in result


def test_execute_empty_with_course_filter_names_course(tool, mock_store):
    mock_store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
    result = tool.execute("query", course_name="MCP Course")
    assert "No relevant content found" in result
    assert "MCP Course" in result


def test_execute_empty_with_lesson_filter_names_lesson(tool, mock_store):
    mock_store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
    result = tool.execute("query", lesson_number=7)
    assert "No relevant content found" in result
    assert "lesson 7" in result.lower()


# ── error path ─────────────────────────────────────────────────────────────

def test_execute_returns_error_string_on_search_failure(tool, mock_store):
    mock_store.search.return_value = SearchResults.empty(
        "Search error: Number of requested results 5 is greater than number of elements in index 0"
    )
    result = tool.execute("query")
    assert "Search error" in result


# ── integration: real VectorStore with a temporary ChromaDB ───────────────

import os
import tempfile


@pytest.fixture(scope="module")
def real_store_with_docs():
    """Populate a temporary ChromaDB with the first real doc file."""
    from vector_store import VectorStore
    from document_processor import DocumentProcessor

    docs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "docs",
    )
    txt_files = [f for f in os.listdir(docs_dir) if f.endswith(".txt")]
    if not txt_files:
        pytest.skip("No .txt files found in docs/")

    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(tmp, "all-MiniLM-L6-v2", max_results=5)
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)
        course, chunks = processor.process_course_document(
            os.path.join(docs_dir, txt_files[0])
        )
        store.add_course_metadata(course)
        store.add_course_content(chunks)
        yield store, course


def test_integration_execute_returns_content(real_store_with_docs):
    store, course = real_store_with_docs
    tool = CourseSearchTool(store)
    result = tool.execute("introduction")
    assert "Search error" not in result, f"Unexpected error: {result}"


def test_integration_execute_with_course_name(real_store_with_docs):
    store, course = real_store_with_docs
    tool = CourseSearchTool(store)
    # Partial course name — _resolve_course_name should handle it
    result = tool.execute("introduction", course_name=course.title[:6])
    assert "Search error" not in result, f"Unexpected error: {result}"


def test_integration_n_results_exceeds_chunk_count(real_store_with_docs):
    """
    ChromaDB raises when n_results > collection size.
    VectorStore.search() should catch this and return SearchResults.empty(),
    not let it propagate — which would cause HTTP 500 / 'query failed'.
    """
    from vector_store import VectorStore
    store, course = real_store_with_docs

    # Create a store pointing at same data but with absurdly large max_results
    greedy_store = VectorStore.__new__(VectorStore)
    greedy_store.course_catalog = store.course_catalog
    greedy_store.course_content = store.course_content
    greedy_store.max_results = 10_000

    results = greedy_store.search("introduction")
    # Must NOT raise; must return a SearchResults (possibly with error)
    assert results is not None
    # If it set an error, surface it so the test output is informative
    if results.error:
        pytest.fail(
            f"search() let the ChromaDB exception propagate as an error string "
            f"instead of gracefully handling it: {results.error}"
        )
