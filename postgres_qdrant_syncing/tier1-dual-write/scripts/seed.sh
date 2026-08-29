#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -f "$SCRIPT_DIR/../.env" ]; then
  export $(grep -v '^#' "$SCRIPT_DIR/../.env" | xargs)
fi

API_URL="${API_URL:-http://localhost:8000}"

echo "Seeding via API at $API_URL ..."
echo "(Make sure the API is running: PYTHONPATH=../.. uvicorn app.main:app --port 8000)"
echo ""

PYTHONPATH="$ROOT_DIR" python -m shared.seed --api "$API_URL"

echo ""
echo "Sync is handled by the dual-write mechanism in each request."
echo "To check: curl -X POST $API_URL/sync/reconcile | jq"
