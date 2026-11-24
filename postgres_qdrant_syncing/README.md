# Postgres ↔ Qdrant Sync

Three architectures for keeping Postgres and Qdrant in sync, in increasing order of complexity. Each tier is a self-contained demo with a FastAPI app, Docker Compose stack, and working search.

---

## Which tier should I use?

| | Tier 1: Dual-Write | Tier 2: Outbox | Tier 3: CDC |
|---|---|---|---|
| **Extra infra** | None | Outbox table + worker | Redpanda + Debezium + consumer |
| **Write path impact** | Qdrant call blocks request | None (async) | None (async) |
| **Qdrant downtime** | Loses sync or fails request | Events queue in Postgres | Events queue in Redpanda |
| **Captures direct SQL** | No | No | Yes |
| **Best for** | Prototypes, MVPs | Most production apps | High throughput, multi-consumer |

**When in doubt, start with Tier 1. Graduate to Tier 2 when you need guaranteed delivery.**

---

## Tiers

### [Tier 1 — Dual Write](./tier1-dual-write/)
Every CRUD request writes to Postgres and Qdrant synchronously. Simple, zero extra infrastructure.

### [Tier 2 — Transactional Outbox](./tier2-outbox/)
Writes go into Postgres atomically alongside an outbox event. A background worker processes events and syncs to Qdrant. Qdrant downtime never blocks writes.

### [Tier 3 — CDC with Debezium](./tier3-cdc/)
The app writes to Postgres only. Debezium reads the WAL and publishes change events to Redpanda. A Python consumer syncs events to Qdrant. Fully decoupled.

---

## Quick Start (any tier)

```bash
# 1. Enter a tier
cd tier1-dual-write   # or tier2-outbox / tier3-cdc

# 2. Configure environment
cp .env.example .env

# 3. Start Postgres + Qdrant (Tier 3: full CDC stack)
docker compose up -d

# 4. Install Python dependencies
pip install -r ../requirements.txt

# 5. Seed Postgres with 1,000 H&M fashion products
bash scripts/seed.sh

# 6. Start the API
PYTHONPATH=.. uvicorn app.main:app --reload --port 8000

# 7. Search
curl "http://localhost:8000/search?q=summer+dress&limit=5"
```

> **Inference:** By default, embeddings are generated locally via `fastembed` — no cloud account needed. Set `CLOUD_INFERENCE=true` in `.env` to use Qdrant Cloud Inference instead.

---

## Dataset

[`Qdrant/hm_ecommerce_products`](https://huggingface.co/datasets/Qdrant/hm_ecommerce_products) — 1,000 H&M fashion products with names, descriptions, colors, departments, and synthetic prices.

## Shared Code

All tiers share [`shared/`](./shared/):
- `postgres.py` — asyncpg CRUD
- `qdrant_helpers.py` — upsert, delete, search
- `seed.py` — loads dataset into Postgres
- `reconcile.py` — detects and fixes drift between Postgres and Qdrant
- `search.py` — hybrid, semantic, and keyword search