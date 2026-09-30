"""
Tier 2: Transactional Outbox FastAPI Application

Start with:
    PYTHONPATH=../.. uvicorn app.main:app --reload --port 8000

The background outbox worker starts automatically with the app.
Set WORKER_MODE=listen to use LISTEN/NOTIFY instead of polling.
"""
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from fastapi import FastAPI

from shared.postgres import close_pool, get_pool, init_schema
from shared.qdrant_helpers import init_collection
from app.routes import router
from app.worker import run_listen_worker, run_polling_worker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WORKER_MODE = os.getenv("WORKER_MODE", "poll")  # "poll" | "listen"

INIT_OUTBOX_SQL = """
CREATE TABLE IF NOT EXISTS sync_outbox (
    id              BIGSERIAL PRIMARY KEY,
    entity_type     VARCHAR(50) NOT NULL DEFAULT 'product',
    entity_id       VARCHAR(20) NOT NULL,
    operation       VARCHAR(10) NOT NULL,
    payload         JSONB,
    status          VARCHAR(20) NOT NULL DEFAULT 'pending',
    attempts        INT NOT NULL DEFAULT 0,
    max_attempts    INT NOT NULL DEFAULT 5,
    last_error      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_outbox_pending
    ON sync_outbox (created_at ASC)
    WHERE status IN ('pending', 'failed');

CREATE OR REPLACE FUNCTION notify_outbox_insert()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify('sync_outbox_insert', NEW.id::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS outbox_notify ON sync_outbox;
CREATE TRIGGER outbox_notify
    AFTER INSERT ON sync_outbox
    FOR EACH ROW EXECUTE FUNCTION notify_outbox_insert();
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_schema()
    pool = await get_pool()
    await pool.execute(INIT_OUTBOX_SQL)
    await init_collection()

    # start the outbox worker as a background task
    if WORKER_MODE == "listen":
        task = asyncio.create_task(run_listen_worker())
    else:
        task = asyncio.create_task(run_polling_worker())
    logger.info("Outbox worker started (mode=%s)", WORKER_MODE)

    yield

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await close_pool()


app = FastAPI(
    title="Postgres <--> Qdrant Sync: Tier 2 — Transactional Outbox",
    description=(
        "Transactional outbox pattern: product writes and outbox events are "
        "committed atomically. A background worker processes events and syncs to Qdrant."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)