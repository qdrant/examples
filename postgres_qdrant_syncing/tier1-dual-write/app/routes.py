"""
Tier 1: Application-Level Dual-Write

Every CRUD endpoint writes to both Postgres and Qdrant sequentially
within the same request handler. Simple, zero extra infra.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query

from shared.models import (
    ProductCreate,
    ProductPatch,
    ProductResponse,
    ProductUpdate,
    ReconcileResult,
    SearchResult,
)
from shared.postgres import (
    delete_product as pg_delete,
    get_product,
    insert_product,
    list_products,
    patch_product,
    update_product,
)
from shared.qdrant_helpers import (
    check_health,
    delete_product as qdrant_delete,
    upsert_product,
)
from shared.reconcile import reconcile
from shared.search import search

logger = logging.getLogger(__name__)

router = APIRouter()


# crud
@router.post("/products", response_model=ProductResponse, status_code=201)
async def create_product(product: ProductCreate):
    # 1. Write to Postgres first — it is the source of truth
    row = await insert_product(product.model_dump())

    # 2. Write to Qdrant — non-blocking on failure; reconcile catches drift
    try:
        await upsert_product(row)
    except Exception as exc:
        logger.error("Qdrant upsert failed for %s: %s", product.article_id, exc)

    return row


@router.get("/products", response_model=list[ProductResponse])
async def list_products_endpoint(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    return await list_products(limit=limit, offset=offset)


@router.get("/products/{article_id}", response_model=ProductResponse)
async def get_product_endpoint(article_id: str):
    row = await get_product(article_id)
    if not row:
        raise HTTPException(status_code=404, detail="Product not found")
    return row


@router.put("/products/{article_id}", response_model=ProductResponse)
async def update_product_endpoint(article_id: str, data: ProductUpdate):
    row = await update_product(article_id, data.model_dump())
    if not row:
        raise HTTPException(status_code=404, detail="Product not found")

    try:
        await upsert_product(row)
    except Exception as exc:
        logger.error("Qdrant upsert failed for %s: %s", article_id, exc)

    return row


@router.patch("/products/{article_id}", response_model=ProductResponse)
async def patch_product_endpoint(article_id: str, data: ProductPatch):
    row = await patch_product(article_id, data.model_dump(exclude_none=True))
    if not row:
        raise HTTPException(status_code=404, detail="Product not found")

    try:
        await upsert_product(row)
    except Exception as exc:
        logger.error("Qdrant upsert failed for %s: %s", article_id, exc)

    return row


@router.delete("/products/{article_id}", status_code=204)
async def delete_product_endpoint(article_id: str):
    deleted = await pg_delete(article_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Product not found")

    try:
        await qdrant_delete(article_id)
    except Exception as exc:
        logger.error("Qdrant delete failed for %s: %s", article_id, exc)


# search routes
@router.get("/search", response_model=list[SearchResult])
async def search_hybrid(
    q: str = Query(..., description="Search query"),
    color: str | None = Query(None),
    product_type: str | None = Query(None),
    limit: int = Query(10, ge=1, le=50),
):
    return await search(q, mode="hybrid", color=color, product_type=product_type, limit=limit)


@router.get("/search/semantic", response_model=list[SearchResult])
async def search_semantic(
    q: str = Query(...),
    color: str | None = Query(None),
    product_type: str | None = Query(None),
    limit: int = Query(10, ge=1, le=50),
):
    return await search(q, mode="semantic", color=color, product_type=product_type, limit=limit)


@router.get("/search/keyword", response_model=list[SearchResult])
async def search_keyword(
    q: str = Query(...),
    color: str | None = Query(None),
    product_type: str | None = Query(None),
    limit: int = Query(10, ge=1, le=50),
):
    return await search(q, mode="keyword", color=color, product_type=product_type, limit=limit)


# ops
@router.get("/health")
async def health():
    from shared.postgres import get_pool

    async def pg_health() -> bool:
        try:
            pool = await get_pool()
            await pool.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    pg_ok, qdrant_ok = await asyncio.gather(pg_health(), check_health())
    return {"postgres": pg_ok, "qdrant": qdrant_ok}


@router.post("/sync/reconcile", response_model=ReconcileResult)
async def reconcile_endpoint(fix: bool = Query(False)):
    return await reconcile(fix=fix)