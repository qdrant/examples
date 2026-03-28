from __future__ import annotations

from decimal import Decimal
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Document, PointStruct

from shared.config import config
from shared.embedding import article_id_to_uuid, build_embedding_text


_client: AsyncQdrantClient | None = None


def get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        kwargs: dict = {
            "url": config.qdrant_url,
            "timeout": 10,
            "cloud_inference": config.cloud_inference,
        }
        if config.qdrant_api_key:
            kwargs["api_key"] = config.qdrant_api_key
        _client = AsyncQdrantClient(**kwargs)
    return _client


async def init_collection() -> None:
    """
    Create the products collection if it doesn't exist.
    """
    client = get_client()
    existing = [c.name for c in (await client.get_collections()).collections]
    if config.qdrant_collection not in existing:
        await client.create_collection(
            collection_name=config.qdrant_collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },
        )

    # Payload indices on commonly filtered fields for better performance
    for field in ["color", "product_type"]:
        try:
            await client.create_payload_index(
                collection_name=config.qdrant_collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            print(f"Warning: Failed to create payload index on '{field}': {exc}")


def _product_to_point(product: dict) -> PointStruct:
    text = build_embedding_text(product)
    point_id = article_id_to_uuid(product["article_id"])

    payload = {
        k: float(v) if isinstance(v, Decimal) else v
        for k, v in product.items()
        if v is not None and k not in ("id", "created_at", "updated_at")
    }
    for dt_field in ("created_at", "updated_at"):
        if product.get(dt_field):
            payload[dt_field] = (
                product[dt_field].isoformat()
                if hasattr(product[dt_field], "isoformat")
                else str(product[dt_field])
            )

    return PointStruct(
        id=point_id,
        payload=payload,
        vector={
            "dense": Document(text=text, model=config.dense_model),
            "bm25": Document(text=text, model=config.sparse_model),
        },
    )


async def upsert_product(product: dict) -> None:
    client = get_client()
    point = _product_to_point(product)
    await client.upsert(
        collection_name=config.qdrant_collection,
        points=[point],
    )


async def upsert_products_batch(products: list[dict], batch_size: int = 100) -> None:
    client = get_client()
    for i in range(0, len(products), batch_size):
        batch = products[i : i + batch_size]
        points = [_product_to_point(p) for p in batch]
        await client.upsert(
            collection_name=config.qdrant_collection,
            points=points,
        )


async def delete_product(article_id: str) -> None:
    client = get_client()
    point_id = article_id_to_uuid(article_id)
    await client.delete(
        collection_name=config.qdrant_collection,
        points_selector=models.PointIdsList(points=[point_id]),
    )


async def get_all_point_ids() -> list[str]:
    """
    Scroll through all points and return article_ids from their payloads.
    """
    client = get_client()
    article_ids = []
    offset = None

    while True:
        results, next_offset = await client.scroll(
            collection_name=config.qdrant_collection,
            limit=500,
            offset=offset,
            with_payload=["article_id"],
            with_vectors=False,
        )
        for point in results:
            if point.payload and "article_id" in point.payload:
                article_ids.append(point.payload["article_id"])
        if next_offset is None:
            break
        offset = next_offset

    return article_ids


def _build_filter(
    color: str | None, product_type: str | None
) -> models.Filter | None:
    conditions = []
    if color:
        conditions.append(
            models.FieldCondition(key="color", match=models.MatchValue(value=color))
        )
    if product_type:
        conditions.append(
            models.FieldCondition(
                key="product_type", match=models.MatchValue(value=product_type)
            )
        )
    return models.Filter(must=conditions) if conditions else None


async def hybrid_search(
    query: str,
    color: str | None = None,
    product_type: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Hybrid search combining dense + BM25 with RRF fusion.
    """
    client = get_client()
    query_filter = _build_filter(color, product_type)

    results = await client.query_points(
        collection_name=config.qdrant_collection,
        prefetch=[
            models.Prefetch(
                query=Document(text=query, model=config.dense_model),
                using="dense",
                limit=limit * 2,
                filter=query_filter,
            ),
            models.Prefetch(
                query=Document(text=query, model=config.sparse_model),
                using="bm25",
                limit=limit * 2,
                filter=query_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )

    return [{**point.payload, "score": point.score} for point in results.points]


async def semantic_search(
    query: str,
    color: str | None = None,
    product_type: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Dense-only semantic search."""
    client = get_client()
    results = await client.query_points(
        collection_name=config.qdrant_collection,
        query=Document(text=query, model=config.dense_model),
        using="dense",
        limit=limit,
        query_filter=_build_filter(color, product_type),
        with_payload=True,
    )
    return [{**point.payload, "score": point.score} for point in results.points]


async def keyword_search(
    query: str,
    color: str | None = None,
    product_type: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    BM25-only keyword search.
    """
    client = get_client()
    results = await client.query_points(
        collection_name=config.qdrant_collection,
        query=Document(text=query, model=config.sparse_model),
        using="bm25",
        limit=limit,
        query_filter=_build_filter(color, product_type),
        with_payload=True,
    )
    return [{**point.payload, "score": point.score} for point in results.points]


async def check_health() -> bool:
    """
    Return True if Qdrant is reachable.
    """
    try:
        await get_client().get_collections()
        return True
    except Exception:
        return False