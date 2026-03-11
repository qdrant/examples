from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from typing import Any

from shared.postgres import get_pool


def _serialize_payload(data: dict) -> str:
    def default(obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    return json.dumps(data, default=default)


async def enqueue_upsert(conn_or_pool, article_id: str, product: dict) -> int:
    """
    Insert a pending upsert event. Pass a connection (inside a transaction)
    or the pool (standalone call).
    """
    row = await conn_or_pool.fetchrow(
        """
        INSERT INTO sync_outbox (entity_id, operation, payload)
        VALUES ($1, 'upsert', $2::jsonb)
        RETURNING id
        """,
        article_id,
        _serialize_payload(product),
    )
    return row["id"]


async def enqueue_delete(conn_or_pool, article_id: str) -> int:
    """
    Insert a pending delete event.
    """
    row = await conn_or_pool.fetchrow(
        """
        INSERT INTO sync_outbox (entity_id, operation, payload)
        VALUES ($1, 'delete', NULL)
        RETURNING id
        """,
        article_id,
    )
    return row["id"]


async def get_sync_status() -> dict:
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE status = 'pending')                         AS pending_events,
            COUNT(*) FILTER (WHERE status = 'failed')                          AS failed_events,
            COUNT(*) FILTER (WHERE status = 'completed'
                             AND processed_at > NOW() - INTERVAL '1 hour')     AS completed_last_hour,
            AVG(EXTRACT(EPOCH FROM (processed_at - created_at)))
                FILTER (WHERE status = 'completed'
                        AND processed_at > NOW() - INTERVAL '1 hour')          AS avg_lag_seconds
        FROM sync_outbox
        """
    )
    return dict(row)