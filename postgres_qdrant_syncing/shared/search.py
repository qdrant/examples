from __future__ import annotations

from typing import Literal

from shared.qdrant_helpers import hybrid_search, keyword_search, semantic_search


async def search(
    query: str,
    mode: Literal["hybrid", "semantic", "keyword"] = "hybrid",
    color: str | None = None,
    product_type: str | None = None,
    limit: int = 10,
) -> list[dict]:
    if mode == "semantic":
        return await semantic_search(query, color=color, product_type=product_type, limit=limit)
    elif mode == "keyword":
        return await keyword_search(query, color=color, product_type=product_type, limit=limit)
    else:
        return await hybrid_search(query, color=color, product_type=product_type, limit=limit)