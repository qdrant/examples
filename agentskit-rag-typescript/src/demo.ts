import { qdrant } from '@agentskit/memory'
import { createRAG } from '@agentskit/rag'
import { embed } from './embed.js'

const qdrantUrl = process.env.QDRANT_URL ?? 'http://localhost:6333'
const collection = process.env.QDRANT_COLLECTION ?? 'agentskit_docs'

async function ensureCollection(): Promise<void> {
  const response = await fetch(`${qdrantUrl}/collections/${collection}`, {
    method: 'PUT',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      vectors: { size: 64, distance: 'Cosine' },
    }),
  })

  if (!response.ok) {
    throw new Error(`Could not create Qdrant collection: ${await response.text()}`)
  }
}

await ensureCollection()

const rag = createRAG({
  embed,
  store: qdrant({
    url: qdrantUrl,
    apiKey: process.env.QDRANT_API_KEY,
    collection,
  }),
  chunkSize: 400,
  chunkOverlap: 40,
  topK: 3,
})

await rag.ingest([
  {
    id: 'agentskit-overview',
    content: 'AgentsKit is a modular TypeScript toolkit for agents, memory, tools, RAG, evaluation, and observability.',
    metadata: { source: 'overview' },
  },
  {
    id: 'qdrant-overview',
    content: 'Qdrant stores and searches high-dimensional vectors with payload metadata and cosine similarity.',
    metadata: { source: 'qdrant' },
  },
])

const results = await rag.search('Which toolkit provides TypeScript RAG and agent memory?')
console.log(results)
