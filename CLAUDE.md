# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
./run.sh
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

Requires a `.env` file in the project root with `ANTHROPIC_API_KEY=...`.

App is served at `http://localhost:8000`. FastAPI auto-docs at `http://localhost:8000/docs`.

On first startup, course documents from `docs/` are automatically indexed into ChromaDB. Re-indexing is skipped for courses already present.

## Architecture

This is a RAG chatbot with a FastAPI backend and a vanilla JS frontend (no build step).

**Request flow:** Browser → `POST /api/query` → `RAGSystem.query()` → Claude API (first call with tool) → `CourseSearchTool` → ChromaDB vector search → Claude API (second call with results) → response.

**Key design decisions:**

- `RAGSystem` (`backend/rag_system.py`) is the central orchestrator. It wires together document processing, vector storage, AI generation, session history, and tool management.
- Claude uses **tool calling** to decide whether to search. The `search_course_content` tool (defined in `search_tools.py`) is passed on every query; Claude may invoke it or answer directly from its own knowledge.
- `AIGenerator` (`backend/ai_generator.py`) handles the two-turn Claude interaction: first call may return `stop_reason="tool_use"`, triggering `_handle_tool_execution()` which runs the tool and makes a second call without tools.
- `VectorStore` (`backend/vector_store.py`) maintains two ChromaDB collections: `course_catalog` (course titles/metadata for fuzzy name resolution) and `course_content` (chunked lesson text for semantic search). Course name matching uses vector similarity against the catalog before filtering content.
- `SessionManager` keeps conversation history in memory (not persisted across restarts), capped at `MAX_HISTORY * 2` messages.

**Document format** (`docs/*.txt`): Files must start with `Course Title:`, `Course Link:`, `Course Instructor:` on the first three lines, followed by `Lesson N: Title` / `Lesson Link:` markers. `DocumentProcessor` chunks lesson text by sentences with configurable overlap.

**Configuration** is centralized in `backend/config.py` via a `Config` dataclass loaded from `.env`.

## Package Management

Uses `uv`. To add dependencies: `uv add <package>`. The `backend/` directory is the working directory for the server — run `uv` commands from the project root.
