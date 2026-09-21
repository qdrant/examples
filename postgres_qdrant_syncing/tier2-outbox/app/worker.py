"""
Outbox worker — processes pending sync_outbox events and syncs them to Qdrant.

Two delivery modes:
  - Polling loop (default): checks every N seconds
  - LISTEN/NOTIFY: wakes up immediately when new events are inserted

Both modes share the same core processing logic and retry semantics.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import asyncpg

from shared.postgres import get_pool
from shared.qdrant_helpers import delete_product as qdrant_delete, upsert_product

logger = logging.getLogger(__name__)

POLL_INTERVAL = float(os.getenv("QDRANT_SYNC_TTL_SECONDS", "2"))
BATCH_SIZE = int(os.getenv("OUTBOX_BATCH_SIZE", "250"))
MAX_ATTEMPTS = int(os.getenv("OUTBOX_MAX_ATTEMPTS", "10"))


# core processing logic (shared by both modes)
async def process_batch() -> int:
    """
    Claim up to BATCH_SIZE pending/failed events, process them, and return
    the number processed.

    Uses FOR UPDATE SKIP LOCKED so multiple workers can run safely in parallel.
    """
    pool = await get_pool()

    async with pool.acquire() as conn:
        # claim a batch atomically
        rows = await conn.fetch(
            """
            UPDATE sync_outbox
            SET status = 'processing', attempts = attempts + 1
            WHERE id IN (
                SELECT id FROM sync_outbox
                WHERE status IN ('pending', 'failed')
                    AND attempts < max_attempts
                ORDER BY created_at ASC
                LIMIT $1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING *
            """,
            BATCH_SIZE,
        )

    if not rows:
        return 0

    logger.debug("Processing %d outbox events", len(rows))

    for row in rows:
        await _process_event(dict(row))

    return len(rows)


async def _process_event(event: dict[str, Any]) -> None:
    pool = await get_pool()

    event_id = event["id"]
    article_id = event["entity_id"]
    operation = event["operation"]

    try:
        if operation == "upsert" and event.get("payload"):
            payload = event["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            await upsert_product(payload)
        elif operation == "delete":
            await qdrant_delete(article_id)
        else:
            logger.warning("Unknown operation '%s' for event %d", operation, event_id)

        # mark completed
        await pool.execute(
            """
            UPDATE sync_outbox
            SET status = 'completed', processed_at = NOW()
            WHERE id = $1
            """,
            event_id,
        )
        logger.debug("Event %d (%s %s) completed", event_id, operation, article_id)

    except Exception as exc:
        logger.error("Event %d failed: %s", event_id, exc)
        async with pool.acquire() as conn:
            # check if we've hit max_attempts
            row = await conn.fetchrow(
                "SELECT attempts, max_attempts FROM sync_outbox WHERE id = $1",
                event_id,
            )
            new_status = "failed" if row["attempts"] >= row["max_attempts"] else "pending"
            await conn.execute(
                """
                UPDATE sync_outbox
                SET status = $2, last_error = $3
                WHERE id = $1
                """,
                event_id,
                new_status,
                str(exc),
            )

# first mode: simple polling loop (default, more robust to missed events)
async def run_polling_worker() -> None:
    """
    Poll the outbox table every POLL_INTERVAL seconds.
    """
    logger.info("Outbox worker started (polling mode, interval=%.1fs)", POLL_INTERVAL)
    while True:
        try:
            processed = await process_batch()
            if processed:
                logger.info("Processed %d outbox events", processed)
        except Exception as exc:
            logger.error("Worker loop error: %s", exc)
        await asyncio.sleep(POLL_INTERVAL)


# second mode: LISTEN/NOTIFY (more immediate but riskier if the worker goes down)
async def run_listen_worker() -> None:
    """
    Listen for NOTIFY signals from the outbox insert trigger and process
    events immediately. Falls back to polling every 30s to catch anything missed.
    """
    logger.info("Outbox worker started (LISTEN/NOTIFY mode)")
    pool = await get_pool()

    # dedicated connection for LISTEN — cannot be a pool connection
    conn: asyncpg.Connection = await asyncpg.connect(pool.get_dsn())

    try:
        await conn.execute("LISTEN sync_outbox_insert")
        logger.info("Listening on channel: sync_outbox_insert")

        while True:
            # wait up to 30s for a notification, then do a sweep anyway
            try:
                notification = await asyncio.wait_for(
                    conn.wait_for_notify(), timeout=30.0
                )
                logger.debug("Received NOTIFY: %s", notification.payload)
            except asyncio.TimeoutError:
                pass

            try:
                processed = await process_batch()
                if processed:
                    logger.info("Processed %d outbox events", processed)
            except Exception as exc:
                logger.error("Worker batch error: %s", exc)
    finally:
        await conn.close()