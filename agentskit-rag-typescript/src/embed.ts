const dimensions = 64

export async function embed(text: string): Promise<number[]> {
  const vector = Array.from({ length: dimensions }, () => 0)
  const tokens = text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []

  for (const token of tokens) {
    let hash = 2166136261
    for (const character of token) {
      hash ^= character.codePointAt(0) ?? 0
      hash = Math.imul(hash, 16777619)
    }
    vector[(hash >>> 0) % dimensions]! += 1
  }

  const magnitude = Math.hypot(...vector)
  return magnitude === 0 ? vector : vector.map(value => value / magnitude)
}
