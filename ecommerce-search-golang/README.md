# E-commerce Vector Search

Semantic product search using Qdrant vector database with cloud inference for embeddings.

## Setup

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env
```

Required environment variables:
- `QDRANT_HOST` - Your Qdrant Cloud host
- `QDRANT_API_KEY` - Your Qdrant API key
- `OPENAI_API_KEY` - OpenAI API key (for embeddings via Qdrant cloud inference)

## Ingest Data

Build and run the ingestion script to populate Qdrant with product data:

```bash
docker build -f Dockerfile.ingest -t ecommerce-ingest .

docker run --rm --env-file .env ecommerce-ingest
```

Optional: Set `NUM_WORKERS` to control parallelism (default: 4).

## Run the App

```bash
docker-compose up --build
```

- Frontend: http://localhost:3000
- API: http://localhost:8080

## Local Development

**Server:**
```bash
source .env
go run ./cmd/server
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```