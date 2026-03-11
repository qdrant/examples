"""
Seed 1,000 H&M products by POSTing to the running FastAPI backend.

Because writes go through the API, the tier's sync mechanism fires naturally:
  - Tier 1: each POST dual-writes to Postgres + Qdrant immediately
  - Tier 2: each POST writes an outbox event; the worker syncs to Qdrant
  - Tier 3: each POST triggers a WAL event; Debezium → Redpanda → consumer syncs to Qdrant

Usage:
    python -m shared.seed                          # uses http://localhost:8000
    python -m shared.seed --api http://localhost:9000
"""
from __future__ import annotations

import asyncio
import random
import sys
from typing import Any

import httpx
from datasets import load_dataset


SAMPLE_SIZE = 5000
CONCURRENCY = 20   # max simultaneous requests — enough to be fast, not overwhelming
DEFAULT_API = "http://localhost:8000"

random.seed(42)


def _map_row(item: dict) -> dict[str, Any]:
    return {
        "article_id": str(item["article_id"]),
        "name": item.get("prod_name", ""),
        "description": item.get("detail_desc") or None,
        "product_type": item.get("product_type_name") or None,
        "product_group": item.get("product_group_name") or None,
        "color": item.get("colour_group_name") or None,
        "department": item.get("department_name") or None,
        "index_name": item.get("index_name") or None,
        "image_url": item.get("image_url") or None,
        "price": round(random.uniform(9.99, 199.99), 2),
    }


async def _post_product(
    client: httpx.AsyncClient,
    product: dict,
    semaphore: asyncio.Semaphore,
    results: dict,
) -> None:
    async with semaphore:
        try:
            r = await client.post("/products", json=product)
            if r.status_code == 201:
                results["inserted"] += 1
            elif r.status_code == 409:
                results["skipped"] += 1
            else:
                results["errors"] += 1
                print(f"  Warning: {product['article_id']} → HTTP {r.status_code}: {r.text[:80]}")
        except Exception as exc:
            results["errors"] += 1
            print(f"  Error posting {product['article_id']}: {exc}")


async def seed(api_url: str = DEFAULT_API) -> dict:
    print(f"Loading {SAMPLE_SIZE} products from HuggingFace dataset...")
    ds = load_dataset("Qdrant/hm_ecommerce_products", split=f"train[:{SAMPLE_SIZE}]")
    products = [_map_row(item) for item in ds]
    print(f"Loaded {len(products)} products.")

    print(f"\nPosting to {api_url}/products (concurrency={CONCURRENCY})...")
    results = {"inserted": 0, "skipped": 0, "errors": 0}
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient(base_url=api_url, timeout=30) as client:
        async with asyncio.TaskGroup() as tg:
            for product in products:
                tg.create_task(_post_product(client, product, semaphore, results))

    print(f"\nDone.")
    print(f"  Inserted : {results['inserted']}")
    print(f"  Skipped  : {results['skipped']}  (already existed)")
    print(f"  Errors   : {results['errors']}")
    return results


if __name__ == "__main__":
    url = DEFAULT_API
    if "--api" in sys.argv:
        idx = sys.argv.index("--api")
        url = sys.argv[idx + 1]
    asyncio.run(seed(api_url=url))