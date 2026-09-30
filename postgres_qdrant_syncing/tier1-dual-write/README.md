# Tier 1: Application-Level Dual-Write

**"Just do it in your app code"**

The simplest approach to keeping Postgres and Qdrant in sync: every CRUD endpoint
writes to both databases sequentially within the same request handler.

## Architecture

```
┌──────────┐     1. Write product     ┌───────────┐
│  Client   │ ───────────────────────► │  FastAPI   │
└──────────┘                           │   App      │
                                       └─────┬─────┘
                                             │
                              ┌──────────────┼──────────────┐
                              │ 2a. INSERT    │              │ 2b. Upsert point
                              ▼                              ▼
                        ┌──────────┐                  ┌──────────┐
                        │ Postgres │                  │  Qdrant  │
                        └──────────┘                  └──────────┘
```

Postgres is always written first (it's the source of truth). Qdrant is written
second and errors are caught — the request doesn't fail if Qdrant is unavailable.

## Quick Start

```bash
# 1. Start Postgres
docker compose up -d

# 2. Configure environment
cp .env.example .env
# Defaults work out of the box — Qdrant runs in Docker with local fastembed inference

# 3. Install dependencies (from repo root)
pip install -r ../../requirements.txt

# 4. Seed Postgres with 1,000 H&M products
bash scripts/seed.sh

# 5. Start the API
PYTHONPATH=../.. uvicorn app.main:app --reload --port 8000

# 6. Try a search
curl "http://localhost:8000/search?q=summer+dress&limit=5"
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/products` | Create product → writes Postgres + Qdrant |
| `GET` | `/products` | List products (from Postgres) |
| `GET` | `/products/{id}` | Get single product |
| `PUT` | `/products/{id}` | Full update → writes Postgres + Qdrant |
| `PATCH` | `/products/{id}` | Partial update → writes Postgres + Qdrant |
| `DELETE` | `/products/{id}` | Delete → removes from Postgres + Qdrant |
| `GET` | `/search?q=...` | Hybrid search (dense + BM25 with RRF) |
| `GET` | `/search/semantic?q=...` | Dense-only semantic search |
| `GET` | `/search/keyword?q=...` | BM25-only keyword search |
| `GET` | `/health` | Postgres + Qdrant connectivity check |
| `POST` | `/sync/reconcile` | Compare Postgres vs. Qdrant, report drift |

## Try It

### Create a product
```bash
curl -s -X POST http://localhost:8000/products \
  -H "Content-Type: application/json" \
  -d '{
    "article_id": "999000001",
    "name": "Linen Summer Dress",
    "description": "Flowy midi dress in breathable linen, perfect for warm weather.",
    "product_type": "Dress",
    "product_group": "Garment Upper body",
    "color": "White",
    "department": "Womens Everyday Collection",
    "price": 59.99
  }' | jq
```

### Get a product
```bash
curl -s http://localhost:8000/products/999000001 | jq
```

### List products (paginated)
```bash
curl -s "http://localhost:8000/products?limit=5&offset=0" | jq
```

### Update a product (full replace)
```bash
curl -s -X PUT http://localhost:8000/products/999000001 \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Linen Summer Dress",
    "description": "Updated: now available in extended sizes.",
    "product_type": "Dress",
    "color": "White",
    "price": 54.99
  }' | jq
```

### Patch a product (partial update — just change the price)
```bash
curl -s -X PATCH http://localhost:8000/products/999000001 \
  -H "Content-Type: application/json" \
  -d '{"price": 49.99}' | jq
```

### Delete a product
```bash
curl -s -X DELETE http://localhost:8000/products/999000001
# Returns 204 No Content on success
```

### Search
```bash
# Hybrid search (dense + BM25 with RRF) — recommended default
curl -s "http://localhost:8000/search?q=summer+dress&limit=5" | jq

# Filter by color
curl -s "http://localhost:8000/search?q=casual+top&color=Black&limit=5" | jq

# Dense-only semantic search
curl -s "http://localhost:8000/search/semantic?q=warm+winter+jacket&limit=5" | jq

# BM25 keyword search
curl -s "http://localhost:8000/search/keyword?q=denim+jeans&limit=5" | jq
```

### Health check
```bash
curl -s http://localhost:8000/health | jq
# {"postgres": true, "qdrant": true}
```

### Check sync status (Postgres vs. Qdrant)
```bash
# Report only
curl -s -X POST "http://localhost:8000/sync/reconcile" | jq

# Auto-fix any drift
curl -s -X POST "http://localhost:8000/sync/reconcile?fix=true" | jq
```

---

## Failure Modes

| Failure | Consequence | Mitigation |
|---------|-------------|------------|
| Qdrant is down | Postgres write succeeds; Qdrant write silently skipped | Logged; `reconcile` fixes drift |
| Qdrant is slow | Request latency spikes (blocks on Qdrant call) | Client has 10s timeout |
| Postgres fails after Qdrant write | Orphaned point in Qdrant | Write Postgres first; `reconcile --fix` cleans up |
| Network partition | Partial writes | Reconciliation script |

## When to Use This

- ✅ Prototypes, MVPs, and internal tools
- ✅ < 10K products, low write throughput
- ✅ Teams that want to ship fast and handle edge cases later
- ❌ When Qdrant downtime should never block writes (→ use Tier 2)
- ❌ When you need guaranteed delivery of every sync event (→ use Tier 2)
- ❌ High write throughput (→ use Tier 2 or Tier 3)

## Reconciliation

Drift can accumulate if Qdrant was briefly unreachable. Run reconciliation to
detect (and optionally fix) differences between Postgres and Qdrant:

```bash
bash scripts/reconcile.sh          # report only
bash scripts/reconcile.sh --fix    # auto-fix: upsert missing, delete orphans
```