"""
Reconciliation: compare Postgres product IDs vs. Qdrant point IDs and report/fix drift.

Usage:
    python -m shared.reconcile          # report only
    python -m shared.reconcile --fix    # auto-fix
"""
from __future__ import annotations

import asyncio
import sys

from shared.models import ReconcileResult
from shared.postgres import get_all_article_ids, get_product
from shared.qdrant_helpers import (
    delete_product as qdrant_delete,
    get_all_point_ids,
    upsert_product,
)


async def reconcile(fix: bool = False) -> ReconcileResult:
    print("Fetching IDs from Postgres...")
    pg_ids = set(await get_all_article_ids())

    print("Fetching IDs from Qdrant...")
    qdrant_ids = set(await get_all_point_ids())

    missing_in_qdrant = pg_ids - qdrant_ids
    orphaned_in_qdrant = qdrant_ids - pg_ids

    result = ReconcileResult(
        postgres_count=len(pg_ids),
        qdrant_count=len(qdrant_ids),
        missing_in_qdrant=len(missing_in_qdrant),
        orphaned_in_qdrant=len(orphaned_in_qdrant),
        in_sync=len(missing_in_qdrant) == 0 and len(orphaned_in_qdrant) == 0,
        missing_ids=sorted(missing_in_qdrant),
        orphaned_ids=sorted(orphaned_in_qdrant),
    )

    print("\n=== Reconciliation Report ===")
    print(f"Postgres products : {result.postgres_count}")
    print(f"Qdrant points     : {result.qdrant_count}")
    print(f"Missing in Qdrant : {result.missing_in_qdrant}")
    print(f"Orphaned in Qdrant: {result.orphaned_in_qdrant}")
    print(f"In sync           : {result.in_sync}")

    if fix and not result.in_sync:
        print("\nApplying fixes...")

        if missing_in_qdrant:
            print(f"  Upserting {len(missing_in_qdrant)} missing products to Qdrant...")
            for article_id in missing_in_qdrant:
                product = await get_product(article_id)
                if product:
                    await upsert_product(product)
            print("  Done upserting.")

        if orphaned_in_qdrant:
            print(f"  Deleting {len(orphaned_in_qdrant)} orphaned points from Qdrant...")
            for article_id in orphaned_in_qdrant:
                await qdrant_delete(article_id)
            print("  Done deleting.")

        print("\nReconciliation complete.")

    return result


if __name__ == "__main__":
    fix = "--fix" in sys.argv
    asyncio.run(reconcile(fix=fix))