-- transactional outbox table for async Qdrant sync
--
-- events are written atomically alongside product writes.
-- a background worker reads from this table and syncs to Qdrant.

CREATE TABLE IF NOT EXISTS sync_outbox (
    id              BIGSERIAL PRIMARY KEY,
    entity_type     VARCHAR(50) NOT NULL DEFAULT 'product',
    entity_id       VARCHAR(20) NOT NULL,           -- article_id
    operation       VARCHAR(10) NOT NULL,            -- 'upsert' | 'delete'
    payload         JSONB,                           -- full product snapshot at write time
    status          VARCHAR(20) NOT NULL DEFAULT 'pending',
    attempts        INT NOT NULL DEFAULT 0,
    max_attempts    INT NOT NULL DEFAULT 5,
    last_error      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at    TIMESTAMPTZ
);

-- index for efficient worker polling: only pending/failed events, ordered by age
CREATE INDEX IF NOT EXISTS idx_outbox_pending
    ON sync_outbox (created_at ASC)
    WHERE status IN ('pending', 'failed');

-- index for status monitoring queries
CREATE INDEX IF NOT EXISTS idx_outbox_status
    ON sync_outbox (status, created_at);

-- optional: notify the worker on new outbox entries
CREATE OR REPLACE FUNCTION notify_outbox_insert()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify('sync_outbox_insert', NEW.id::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS outbox_notify ON sync_outbox;
CREATE TRIGGER outbox_notify
    AFTER INSERT ON sync_outbox
    FOR EACH ROW EXECUTE FUNCTION notify_outbox_insert();