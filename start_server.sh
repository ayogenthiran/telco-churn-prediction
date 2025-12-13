#!/bin/bash
# Script to start the FastAPI server with the correct Python version

cd "$(dirname "$0")"

echo "🚀 Starting FastAPI server..."
echo "📍 Server will be available at: http://127.0.0.1:8000"
echo "📚 API docs: http://127.0.0.1:8000/docs"
echo "🎨 Gradio UI: http://127.0.0.1:8000/ui"
echo ""
echo "Press CTRL+C to stop the server"
echo ""

python -m uvicorn src.app.main:app --reload --host 127.0.0.1 --port 8000

