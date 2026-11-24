# Tier 2: Transactional Outbox Pattern

**"Never lose a sync event"**

Every write to Postgres also inserts an event into a `sync_outbox` table **in the
same transaction**. A background worker processes these events and syncs them to
Qdrant asynchronously. If Qdrant is down, events simply queue up in Postgres until
it recovers — your write path is never blocked.

## Architecture

```
┌──────────┐     1. Write product     ┌───────────┐
│  Client   │ ───────────────────────► │  FastAPI   │
└──────────┘                           │   App      │
                                       └─────┬─────┘
                                             │
                              2. Single Postgres transaction:
                              ┌──────────────────────────────┐
                              │  INSERT INTO products (...)   │
                              │  INSERT INTO sync_outbox (...) │
                              └──────────────┬───────────────┘
                                             │
                                      ┌──────────┐
                                      │ Postgres │
                                      └─────┬────┘
                                            │
                             3. Worker: poll or LISTEN/NOTIFY
                                            │
                                            ▼
                                      ┌───────────┐
                                      │  Worker   │
                                      └─────┬─────┘
                                            │
                             4. Upsert/delete in Qdrant
                                            ▼
                                      ┌──────────┐
                                      │  Qdrant  │
                                      └──────────┘
```

## Quick Start

```bash
# 1. Start Postgres
docker compose up -d

# 2. Configure environment
cp .env.example .env
# Edit .env — set QDRANT_URL and QDRANT_API_KEY

# 3. Install dependencies (from repo root)
pip install -r ../../requirements.txt

# 4. Seed Postgres
bash scripts/seed.sh

# 5. Start the API (worker starts automatically)
PYTHONPATH=../.. uvicorn app.main:app --reload --port 8000

# 6. Check worker sync status
curl http://localhost:8000/sync/status

# 7. Try a search (give the worker a few seconds to sync)
curl "http://localhost:8000/search?q=summer+dress&limit=5"
```

## Worker Modes

Set `WORKER_MODE` in `.env`:

| Mode | Env value | Behaviour |
|------|-----------|-----------|
| Polling (default) | `poll` | Checks outbox every `QDRANT_SYNC_TTL_SECONDS` (default 2s) |
| LISTEN/NOTIFY | `listen` | Wakes instantly on new events via Postgres `NOTIFY`; falls back to 30s sweep |

## Outbox Table

```sql
CREATE TABLE sync_outbox (
    id           BIGSERIAL PRIMARY KEY,
    entity_id    VARCHAR(20) NOT NULL,   -- article_id
    operation    VARCHAR(10) NOT NULL,   -- 'upsert' | 'delete'
    payload      JSONB,                  -- product snapshot at write time
    status       VARCHAR(20) DEFAULT 'pending',
    attempts     INT DEFAULT 0,
    max_attempts INT DEFAULT 5,
    last_error   TEXT,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);
```

## API

Same endpoints as Tier 1, plus:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/sync/status` | Pending/failed event counts and avg lag |
| `POST` | `/sync/reconcile` | Full Postgres vs. Qdrant reconciliation |

## Failure Modes

| Failure | Consequence | Mitigation |
|---------|-------------|------------|
| Qdrant is down | Events queue in `sync_outbox` | Worker retries with backoff on recovery |
| Worker crashes | Events remain `pending` | Worker picks up on restart (`SKIP LOCKED` prevents double-processing) |
| Duplicate processing | Same event processed twice | Qdrant upserts are idempotent by point ID |
| Outbox table grows | Storage/perf impact | Prune `completed` events older than N days |