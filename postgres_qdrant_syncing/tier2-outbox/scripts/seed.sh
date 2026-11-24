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
echo "Products are in Postgres. The outbox worker will sync them to Qdrant."
echo "To check worker status: curl $API_URL/sync/status | jq"
echo "To check sync: curl -X POST $API_URL/sync/reconcile | jq"