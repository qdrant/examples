#!/usr/bin/env bash
# Register the Debezium Postgres connector via the Kafka Connect REST API.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONNECT_URL="${DEBEZIUM_CONNECT_URL:-http://localhost:8083}"
CONFIG_FILE="$SCRIPT_DIR/../debezium/connector-config.json"

echo "Waiting for Debezium Connect to be ready..."
until curl -sf "$CONNECT_URL/connectors" > /dev/null; do
  echo "  Not ready yet, retrying in 5s..."
  sleep 5
done
echo "Debezium Connect is ready."

echo ""
echo "Registering connector from: $CONFIG_FILE"
RESPONSE=$(curl -sf -X POST \
  -H "Content-Type: application/json" \
  --data @"$CONFIG_FILE" \
  "$CONNECT_URL/connectors")

echo "Connector registered:"
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"

echo ""
echo "To check connector status:"
echo "  curl $CONNECT_URL/connectors/products-connector/status"