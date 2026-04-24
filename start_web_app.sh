#!/bin/bash

echo "=========================================="
echo "🐕 Dog Emotion Recognition Web App"
echo "=========================================="
echo ""

# Project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

# =========================================================
# Check model files
# =========================================================

echo "📦 Checking model files..."

if [ ! -f "$PROJECT_ROOT/best_models/detection_YOLOv8_baseline.pt" ]; then
    echo "❌ Detection model not found"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/best_models/emotion_ResNet50_baseline.pth" ]; then
    echo "❌ Emotion model not found"
    exit 1
fi

echo "✅ Model files found"
echo ""

# =========================================================
# Start Backend
# =========================================================

echo "🚀 Starting Backend API..."

cd "$PROJECT_ROOT/api_service"

# Activate venv if exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install dependencies if needed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
fi

# IMPORTANT:
# Must use 0.0.0.0 for SageMaker proxy
echo "Starting FastAPI on port 8000..."

uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload &

BACKEND_PID=$!

echo "Backend PID: $BACKEND_PID"

echo "Waiting for backend startup..."
sleep 5

# Health check
if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "✅ Backend started successfully"
else
    echo "⚠️ Backend may not have started properly"
fi

echo ""

# =========================================================
# Start Frontend
# =========================================================

echo "🎨 Starting Frontend..."

cd "$PROJECT_ROOT/web_intf"

# Install npm packages if needed
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi

echo "Starting Vite dev server..."

npm run dev &

FRONTEND_PID=$!

echo "Frontend PID: $FRONTEND_PID"

echo ""

# =========================================================
# URLs
# =========================================================

echo "=========================================="
echo "✅ Application Started!"
echo "=========================================="
echo ""

echo "LOCAL URLS INSIDE CONTAINER:"
echo "Backend API:  http://127.0.0.1:8000"
echo "Frontend App: http://127.0.0.1:5173"
echo ""

echo "SAGEMAKER PROXY URL:"
echo ""
echo "Frontend:"
echo "https://YOUR_DOMAIN/jupyterlab/default/proxy/5173/"
echo ""

echo "Backend Docs:"
echo "https://YOUR_DOMAIN/jupyterlab/default/proxy/8000/docs"
echo ""

echo "Press Ctrl+C to stop all services"
echo "=========================================="

# =========================================================
# Cleanup
# =========================================================

cleanup() {

    echo ""
    echo "🛑 Stopping services..."

    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null

    echo "✅ All services stopped"

    exit 0
}

trap cleanup SIGINT

wait