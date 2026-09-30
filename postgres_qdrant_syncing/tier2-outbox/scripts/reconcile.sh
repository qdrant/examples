#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -f "$SCRIPT_DIR/../.env" ]; then
  export $(grep -v '^#' "$SCRIPT_DIR/../.env" | xargs)
fi

FIX=${1:-""}

if [ "$FIX" = "--fix" ]; then
  PYTHONPATH="$ROOT_DIR" python -m shared.reconcile --fix
else
  PYTHONPATH="$ROOT_DIR" python -m shared.reconcile
fi