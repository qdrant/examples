-- Products table — the source of truth for all tiers

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