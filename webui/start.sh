#!/bin/bash
cd "$(dirname "$0")"

echo "Starting Bernini WebUI..."

# --- Environment ---------------------------------------------------------
# Hugging Face cache lives on the big volume (home only has ~50GB free).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# This box is a shared 8x A100 node; the idle GPU changes over time. The backend
# uses cuda:0, so we pin it (via CUDA_VISIBLE_DEVICES) to whichever physical GPU
# currently has the most free memory. Override by exporting CUDA_VISIBLE_DEVICES
# yourself before running this script.
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free \
        --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi
export CUDA_VISIBLE_DEVICES

# Kill existing processes
pkill -f "uvicorn.*server:app" 2>/dev/null
pkill -f "vite.*frontend" 2>/dev/null

# Start backend
echo "Starting backend on port 8000 (HF_HOME=$HF_HOME, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)..."
cd backend
source ../../.venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait for backend
sleep 2

# Start frontend
echo "Starting frontend on port 5173..."
cd frontend
npm run dev -- --host 0.0.0.0 &
FRONTEND_PID=$!
cd ..

echo ""
echo "=========================================="
echo "Bernini WebUI is running!"
echo "Frontend: http://localhost:5173"
echo "Backend:  http://localhost:8000"
echo "=========================================="
echo ""
echo "Press Ctrl+C to stop"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
