from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from shared.models import (
    ProductCreate,
    ProductPatch,
    ProductResponse,
    ProductUpdate,
    ReconcileResult,
    SearchResult,
    SyncStatus,
)
from shared.postgres import (
    get_pool,
    get_product,
    list_products,
    patch_product,
)
from shared.qdrant_helpers import check_health
from shared.reconcile import reconcile
from shared.search import search
from app.outbox import enqueue_delete, enqueue_upsert, get_sync_status

logger = logging.getLogger(__name__)

router = APIRouter()

# crud stuff
@router.post("/products", response_model=ProductResponse, status_code=201)
async def create_product(product: ProductCreate):
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # first, insert product
            row = await conn.fetchrow(
                """
                INSERT INTO products (
                    article_id, name, description, product_type, product_group,
                    color, department, index_name, image_url, price
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                RETURNING *
                """,
                product.article_id, product.name, product.description,
                product.product_type, product.product_group, product.color,
                product.department, product.index_name, product.image_url,
                product.price,
            )
            # then, enqueue for syncing
            await enqueue_upsert(conn, product.article_id, dict(row))
    return dict(row)


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
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                UPDATE products SET
                    name=$2, description=$3, product_type=$4, product_group=$5,
                    color=$6, department=$7, index_name=$8, image_url=$9, price=$10
                WHERE article_id=$1
                RETURNING *
                """,
                article_id, data.name, data.description, data.product_type,
                data.product_group, data.color, data.department, data.index_name,
                data.image_url, data.price,
            )
            if not row:
                raise HTTPException(status_code=404, detail="Product not found")
            await enqueue_upsert(conn, article_id, dict(row))
    return dict(row)


@router.patch("/products/{article_id}", response_model=ProductResponse)
async def patch_product_endpoint(article_id: str, data: ProductPatch):
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await patch_product(article_id, data.model_dump(exclude_none=True))
            if not row:
                raise HTTPException(status_code=404, detail="Product not found")
            await enqueue_upsert(conn, article_id, row)
    return row


@router.delete("/products/{article_id}", status_code=204)
async def delete_product_endpoint(article_id: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            result = await conn.execute(
                "DELETE FROM products WHERE article_id = $1", article_id
            )
            if result == "DELETE 0":
                raise HTTPException(status_code=404, detail="Product not found")
            await enqueue_delete(conn, article_id)

# search endpoints
@router.get("/search", response_model=list[SearchResult])
async def search_hybrid(
    q: str = Query(...),
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


# pure ops
@router.get("/health")
async def health():
    from shared.postgres import get_pool
    try:
        pool = await get_pool()
        await pool.fetchval("SELECT 1")
        pg_ok = True
    except Exception:
        pg_ok = False
    qdrant_ok = await check_health()
    return {"postgres": pg_ok, "qdrant": qdrant_ok}


@router.get("/sync/status", response_model=SyncStatus)
async def sync_status():
    return await get_sync_status()


@router.post("/sync/reconcile", response_model=ReconcileResult)
async def reconcile_endpoint(fix: bool = Query(False)):
    return await reconcile(fix=fix)