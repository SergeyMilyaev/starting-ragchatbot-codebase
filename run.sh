#!/bin/bash

mkdir -p docs

if [ ! -d "backend" ]; then
    echo "Error: backend directory not found"
    exit 1
fi

PROVIDER="${1:-anthropic}"

echo "Starting Course Materials RAG System..."
echo "Provider: $PROVIDER"
echo "Make sure you have set your API key in .env (ANTHROPIC_API_KEY or GOOGLE_API_KEY)"

uv run python main.py --provider "$PROVIDER" --reload
