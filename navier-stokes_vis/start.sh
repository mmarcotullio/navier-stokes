ROOT="$(cd "$(dirname "$0")" && pwd)"

cleanup() {
  echo ""
  echo "Shutting down..."
  exec 2>/dev/null
  kill 0
  exit 0
}
trap cleanup INT TERM EXIT

# ── Start main_vis silently in background ─────────────────────────────────────
"$ROOT/main_vis/start.sh" > /dev/null 2>&1 &

# ── Wait for main_vis Vite (port 5173) ───────────────────────────────────────
for i in $(seq 1 60); do
  if curl -sf http://localhost:5173 > /dev/null 2>&1; then
    break
  fi
  sleep 1
done

# ── Start story app ───────────────────────────────────────────────────────────
cd "$ROOT/story/app"
npm run dev -- --port 5174

wait
