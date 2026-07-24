import { describe, expect, it } from 'vitest'
import { embed } from '../src/embed.js'

describe('local example embedder', () => {
  it('is deterministic and normalized', async () => {
    const [first, second] = await Promise.all([
      embed('AgentsKit Qdrant RAG'),
      embed('AgentsKit Qdrant RAG'),
    ])

    expect(first).toEqual(second)
    expect(first).toHaveLength(64)
    expect(Math.hypot(...first!)).toBeCloseTo(1)
  })

  it('puts related text closer than unrelated text', async () => {
    const [query, related, unrelated] = await Promise.all([
      embed('typescript rag memory'),
      embed('agentskit typescript rag memory'),
      embed('cooking pasta recipe'),
    ])
    const similarity = (left: number[], right: number[]) =>
      left.reduce((sum, value, index) => sum + value * right[index]!, 0)

    expect(similarity(query, related)).toBeGreaterThan(similarity(query, unrelated))
  })
})
