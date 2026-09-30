"""
Tier 1: Dual-Write FastAPI Application

Start with:
    PYTHONPATH=.. uvicorn app.main:app --reload --port 8000
"""
import sys
import os

# allow importing from the repo root (shared/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from fastapi import FastAPI

from shared.postgres import close_pool, init_schema
from shared.qdrant_helpers import init_collection
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_schema()
    await init_collection()
    yield
    await close_pool()


app = FastAPI(
    title="Postgres ↔ Qdrant Sync: Tier 1 — Dual Write",
    description=(
        "Application-level dual-write: every CRUD operation writes to both "
        "Postgres and Qdrant synchronously within the request handler."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
