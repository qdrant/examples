# AgentsKit RAG with Qdrant

This example builds a small retrieval-augmented generation pipeline in TypeScript with
[`@agentskit/rag`](https://www.npmjs.com/package/@agentskit/rag) and Qdrant-backed vector
memory from [`@agentskit/memory`](https://www.npmjs.com/package/@agentskit/memory).

The included embedder is local, deterministic, and credential-free so the complete
ingest/retrieve flow is easy to reproduce. Replace `src/embed.ts` with any production
embedding provider without changing the RAG or Qdrant integration.

## Run locally

Start Qdrant:

```bash
docker run --rm -p 6333:6333 qdrant/qdrant:v1.15.4
```

In another terminal:

```bash
npm install
npm run demo
```

To use Qdrant Cloud, provide the cluster URL and API key:

```bash
QDRANT_URL="https://your-cluster.cloud.qdrant.io" \
QDRANT_API_KEY="your-api-key" \
npm run demo
```

## What the example demonstrates

- creating a cosine collection through the Qdrant REST API;
- chunking and ingesting documents with AgentsKit;
- preserving source IDs and payload metadata in Qdrant;
- retrieving ranked context through the standard AgentsKit `Retriever` contract;
- swapping the embedder independently of the vector database.

Run `npm test` for the credential-free embedder checks and `npm run check` for strict
TypeScript validation.
