import argparse
import os
import sys
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="RAG Chatbot — start the FastAPI server")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "gemini"],
        default="anthropic",
        help="AI provider to use (default: anthropic)",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", default=False, help="Enable auto-reload")
    args = parser.parse_args()

    # Set before uvicorn imports app.py so RAGSystem sees it at construction time
    os.environ["AI_PROVIDER"] = args.provider

    print(f"Starting RAG Chatbot with provider: {args.provider}")

    backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
    os.chdir(backend_dir)
    sys.path.insert(0, backend_dir)

    uvicorn.run("app:app", host="0.0.0.0", port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
