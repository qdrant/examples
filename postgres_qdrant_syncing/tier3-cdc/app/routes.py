"""
Tier 3: CDC — Pure Postgres CRUD

The application only writes to Postgres. There is no Qdrant awareness here.
Debezium reads the WAL and publishes events to Redpanda; a separate consumer
service syncs those events to Qdrant.
"""
from __future__ import annotations

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
    delete_product,
    get_product,
    insert_product,
    list_products,
    patch_product,
    update_product,
)
from shared.qdrant_helpers import check_health
from shared.reconcile import reconcile
from shared.search import search

router = APIRouter()


# crud -- no awareness of Qdrant or search functionality
@router.post("/products", response_model=ProductResponse, status_code=201)
async def create_product(product: ProductCreate):
    return await insert_product(product.model_dump())


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
    return row


@router.patch("/products/{article_id}", response_model=ProductResponse)
async def patch_product_endpoint(article_id: str, data: ProductPatch):
    row = await patch_product(article_id, data.model_dump(exclude_none=True))
    if not row:
        raise HTTPException(status_code=404, detail="Product not found")
    return row


@router.delete("/products/{article_id}", status_code=204)
async def delete_product_endpoint(article_id: str):
    deleted = await delete_product(article_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Product not found")


# search only
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


@router.post("/sync/reconcile", response_model=ReconcileResult)
async def reconcile_endpoint(fix: bool = Query(False)):
    return await reconcile(fix=fix)