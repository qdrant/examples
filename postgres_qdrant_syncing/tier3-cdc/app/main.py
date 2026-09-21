"""
Tier 3: CDC FastAPI Application

The app writes to Postgres only — no Qdrant, no outbox.
Debezium + Redpanda handle the event pipeline; the consumer service syncs to Qdrant.

Start with:
    PYTHONPATH=../.. uvicorn app.main:app --reload --port 8000
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from fastapi import FastAPI

from shared.postgres import close_pool, init_schema
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_schema()
    yield
    await close_pool()


app = FastAPI(
    title="Postgres <--> Qdrant Sync: Tier 3 — CDC with Debezium",
    description=(
        "Pure Postgres CRUD. Debezium reads the WAL and publishes change events "
        "to Redpanda. A separate consumer service syncs events to Qdrant."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)