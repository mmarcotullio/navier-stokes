#!/usr/bin/env bash
# Start both the Flask inference backend and the Vite dev server.

set -e
VIS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
THREE_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$THREE_DIR/app"

# Kill any stale instances from a previous run
echo "==> Clearing stale processes on ports 5050 / 5173 ..."
fuser -k 5050/tcp 2>/dev/null || true
fuser -k 5173/tcp 2>/dev/null || true
pkill -f "server.py" 2>/dev/null || true
sleep 0.5

# Activate the project virtual environment
source "$VIS_DIR/venv/bin/activate"
export VIS_DIR  # available to server.py if needed

echo "==> Starting Flask backend on http://localhost:5050 ..."
cd "$THREE_DIR"
python server.py &
FLASK_PID=$!

echo "==> Waiting for Flask to be ready..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:5050/api/health >/dev/null 2>&1; then
        echo "==> Flask is ready."
        break
    fi
    sleep 1
done

echo "==> Starting Vite dev server on http://localhost:5173 ..."
cd "$APP_DIR"
npm run dev &
VITE_PID=$!

trap "echo 'Shutting down...'; kill $FLASK_PID $VITE_PID 2>/dev/null" EXIT INT TERM

echo ""
echo "  Open:  http://localhost:5173"
echo ""
wait
