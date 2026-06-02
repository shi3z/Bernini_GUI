#!/bin/bash
cd "$(dirname "$0")"

echo "Starting Bernini WebUI..."

# Kill existing processes
pkill -f "uvicorn.*server:app" 2>/dev/null
pkill -f "vite.*frontend" 2>/dev/null

# Start backend
echo "Starting backend on port 8000..."
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
