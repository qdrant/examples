"""
Postgres connection pool and CRUD operations using asyncpg.
"""
from __future__ import annotations

import asyncpg

from shared.config import config


_pool: asyncpg.Pool | None = None

CREATE_PRODUCTS_TABLE = """
CREATE TABLE IF NOT EXISTS products (
    id              SERIAL PRIMARY KEY,
    article_id      VARCHAR(20) UNIQUE NOT NULL,
    name            VARCHAR(255) NOT NULL,
    description     TEXT,
    product_type    VARCHAR(100),
    product_group   VARCHAR(100),
    color           VARCHAR(50),
    department      VARCHAR(100),
    index_name      VARCHAR(100),
    image_url       VARCHAR(500),
    price           DECIMAL(10,2),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS products_updated_at ON products;
CREATE TRIGGER products_updated_at
    BEFORE UPDATE ON products
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
"""


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=config.asyncpg_dsn,
            min_size=2,
            max_size=10,
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def init_schema() -> None:
    pool = await get_pool()
    await pool.execute(CREATE_PRODUCTS_TABLE)


async def insert_product(product: dict) -> dict:
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        INSERT INTO products (
            article_id, name, description, product_type, product_group,
            color, department, index_name, image_url, price
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
        )
        RETURNING *
        """,
        product["article_id"],
        product["name"],
        product.get("description"),
        product.get("product_type"),
        product.get("product_group"),
        product.get("color"),
        product.get("department"),
        product.get("index_name"),
        product.get("image_url"),
        product.get("price"),
    )
    return dict(row)


async def get_product(article_id: str) -> dict | None:
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT * FROM products WHERE article_id = $1", article_id
    )
    return dict(row) if row else None


async def list_products(limit: int = 20, offset: int = 0) -> list[dict]:
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT * FROM products ORDER BY created_at DESC LIMIT $1 OFFSET $2",
        limit,
        offset,
    )
    return [dict(r) for r in rows]


async def update_product(article_id: str, data: dict) -> dict | None:
    """
    Full replacement update.
    """
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        UPDATE products SET
            name = $2, description = $3, product_type = $4, product_group = $5,
            color = $6, department = $7, index_name = $8, image_url = $9, price = $10
        WHERE article_id = $1
        RETURNING *
        """,
        article_id,
        data["name"],
        data.get("description"),
        data.get("product_type"),
        data.get("product_group"),
        data.get("color"),
        data.get("department"),
        data.get("index_name"),
        data.get("image_url"),
        data.get("price"),
    )
    return dict(row) if row else None


async def patch_product(article_id: str, data: dict) -> dict | None:
    """
    Partial update — only sets provided fields.
    """
    # build dynamic SET clause
    fields = {k: v for k, v in data.items() if v is not None}
    if not fields:
        return await get_product(article_id)

    set_parts = []
    values = [article_id]
    for i, (col, val) in enumerate(fields.items(), start=2):
        set_parts.append(f"{col} = ${i}")
        values.append(val)

    sql = f"UPDATE products SET {', '.join(set_parts)} WHERE article_id = $1 RETURNING *"
    pool = await get_pool()
    row = await pool.fetchrow(sql, *values)
    return dict(row) if row else None


async def delete_product(article_id: str) -> bool:
    pool = await get_pool()
    result = await pool.execute(
        "DELETE FROM products WHERE article_id = $1", article_id
    )
    return result == "DELETE 1"


async def get_all_article_ids() -> list[str]:
    pool = await get_pool()
    rows = await pool.fetch("SELECT article_id FROM products")
    return [r["article_id"] for r in rows]