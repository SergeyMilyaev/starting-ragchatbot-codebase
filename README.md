# Course Materials RAG System

A Retrieval-Augmented Generation (RAG) system designed to answer questions about course materials using semantic search and AI-powered responses.

## Overview

This application is a full-stack web application that enables users to query course materials and receive intelligent, context-aware responses. It uses ChromaDB for vector storage, supports Google Gemini and Anthropic Claude for AI generation, and provides a web interface for interaction.

## AI Provider Support

The system supports two AI providers, selectable via the `AI_PROVIDER` environment variable:

| Provider | Value | Model configured in `config.py` |
|---|---|---|
| Google Gemini | `gemini` | `GEMINI_MODEL` |
| Anthropic Claude | `anthropic` (default) | `ANTHROPIC_MODEL` |

Both providers use the same tool-calling pipeline (`search_course_content`, `get_course_outline`) and produce identical response behaviour.

## Prerequisites

- Python 3.13 or higher
- uv (Python package manager)
- A Google Gemini API key **or** an Anthropic API key (or both)
- **For Windows**: Use Git Bash to run the application commands - [Download Git for Windows](https://git-scm.com/downloads/win)

## Installation

1. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Python dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**

   Create a `.env` file in the root directory. Include the key(s) for the provider(s) you intend to use:
   ```bash
   # For Gemini (set AI_PROVIDER=gemini when running)
   GOOGLE_API_KEY=your_google_api_key_here

   # For Anthropic Claude (default provider)
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

## Running the Application

### Quick Start

Use the provided shell script:
```bash
chmod +x run.sh
./run.sh
```

### Manual Start

```bash
cd backend
uv run uvicorn app:app --reload --port 8000
```

### Selecting the AI provider

Pass `AI_PROVIDER` before the command (or add it to your `.env`):

```bash
# Use Google Gemini
AI_PROVIDER=gemini uv run uvicorn app:app --reload --port 8000

# Use Anthropic Claude (default)
AI_PROVIDER=anthropic uv run uvicorn app:app --reload --port 8000
```

The application will be available at:
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

## Running Tests

The test suite covers the Gemini AI provider path, the search tools, and integration with ChromaDB. No API key is required — the AI generator is mocked.

```bash
uv run python -m pytest backend/tests/ -v
```

### Test coverage

| File | What it tests |
|---|---|
| `backend/tests/test_gemini_generator.py` | Gemini API calls, Anthropic→Gemini tool format conversion, tool execution flow, system prompt completeness |
| `backend/tests/test_rag_system.py` | Gemini RAG pipeline, source lifecycle, session history, integration with real ChromaDB |
| `backend/tests/test_search_tool.py` | `CourseSearchTool.execute()` — result formatting, filters, source tracking, error handling |

