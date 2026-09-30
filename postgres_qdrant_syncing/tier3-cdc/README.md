# Tier 3: Change Data Capture (CDC) with Debezium + Redpanda

**"Let Postgres tell Qdrant what changed"**

The application writes to Postgres only — no dual-write, no outbox table. Debezium
reads Postgres's WAL (Write-Ahead Log) and publishes every INSERT, UPDATE, and DELETE
as a structured event to Redpanda. A Python consumer service reads those events and
syncs them to Qdrant.

## Architecture

```
┌──────────┐     1. Write product     ┌───────────┐
│  Client   │ ───────────────────────► │  FastAPI   │
└──────────┘                           │   App      │
                                       └─────┬─────┘
                                             │
                              2. Normal INSERT/UPDATE/DELETE
                                             │
                                             ▼
                                       ┌──────────┐
                                       │ Postgres │
                                       │   WAL    │ ◄── logical replication slot
                                       └─────┬────┘
                                             │
                              3. Debezium reads WAL changes
                                             ▼
                                  ┌─────────────────────┐
                                  │  Debezium (Connect)  │
                                  └──────────┬──────────┘
                                             │
                              4. Publishes change events
                                             ▼
                                  ┌─────────────────────┐
                                  │      Redpanda        │
                                  │ topic: pgserver.     │
                                  │        public.       │
                                  │        products      │
                                  └──────────┬──────────┘
                                             │
                              5. Consumer reads events
                                             ▼
                                  ┌─────────────────────┐
                                  │   Python Consumer    │
                                  └──────────┬──────────┘
                                             │
                              6. Upsert/delete in Qdrant
                                             ▼
                                       ┌──────────┐
                                       │  Qdrant  │
                                       └──────────┘
```

## Quick Start

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env — set QDRANT_URL and QDRANT_API_KEY

# 2. Start the full stack (Postgres + Redpanda + Debezium + Consumer)
docker compose up -d

# 3. Register the Debezium connector (wait ~30s for services to start)
bash scripts/setup-connector.sh

# 4. Seed Postgres (Debezium snapshot will capture existing rows)
bash scripts/seed.sh

# 5. Start the FastAPI app
PYTHONPATH=../.. uvicorn app.main:app --reload --port 8000

# 6. Watch the consumer sync events to Qdrant
docker compose logs -f consumer

# 7. Try a search (give the consumer a few seconds)
curl "http://localhost:8000/search?q=summer+dress&limit=5"
```

## Components

| Component | Role |
|-----------|------|
| **Postgres** | Source of truth; WAL enabled (`wal_level=logical`) |
| **Debezium** | Reads WAL via logical replication slot; publishes events to Redpanda |
| **Redpanda** | Kafka-compatible event bus (single binary, no ZooKeeper) |
| **Python consumer** | Reads events from Redpanda; upserts/deletes in Qdrant |
| **FastAPI app** | Pure Postgres CRUD; no Qdrant or sync awareness |

> **Redpanda vs Kafka:** Redpanda uses the Kafka wire protocol so the consumer
> works with full Apache Kafka too. Swap `KAFKA_BOOTSTRAP_SERVERS` to point at
> your Kafka cluster and everything works unchanged.

## Connector Management

```bash
# Register connector
bash scripts/setup-connector.sh

# Check status
curl http://localhost:8083/connectors/products-connector/status

# Restart connector (after failures)
curl -X POST http://localhost:8083/connectors/products-connector/restart

# Delete connector (to reset)
curl -X DELETE http://localhost:8083/connectors/products-connector
```

## API

Same endpoints as Tier 1, plus:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/sync/reconcile` | Compare Postgres vs. Qdrant, report drift |

## Failure Modes

| Failure | Consequence | Mitigation |
|---------|-------------|------------|
| Qdrant is down | Consumer pauses; Redpanda retains events | Consumer retries; resume from offset on recovery |
| Redpanda is down | Debezium buffers in WAL | Monitor WAL size; Redpanda HA in production |
| Debezium crashes | WAL retains changes since last checkpoint | Debezium resumes from replication slot on restart |
| Schema change | Debezium connector may need restart | Monitor connector status; test migrations in staging |

## When to Use This

- ✅ High write throughput systems
- ✅ When multiple services need to react to data changes
- ✅ Application code should be completely decoupled from sync
- ✅ Teams with existing Redpanda/Kafka infrastructure
- ✅ When you need to replay history or rebuild the Qdrant index from scratch
- ❌ Simple apps with moderate write volume (→ use Tier 2)
- ❌ When operational complexity must be minimal (→ use Tier 1 or 2)

## Important Notes

- Postgres must run with `wal_level=logical` (set in `docker-compose.yml`)
- Monitor the replication slot lag to prevent WAL disk bloat:
  ```sql
  SELECT slot_name, pg_size_pretty(pg_wal_lsn_diff(pg_current_wal_lsn(), confirmed_flush_lsn))
  FROM pg_replication_slots;
  ```
- The consumer uses at-least-once delivery; Qdrant upserts are idempotent