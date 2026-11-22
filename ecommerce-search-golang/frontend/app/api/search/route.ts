import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8080'

export interface SearchParams {
  query: string
  limit?: number
}

export async function POST(request: NextRequest) {
  try {
    const startTime = performance.now()

    const body: SearchParams = await request.json()
    const { query, limit } = body

    const response = await fetch(`${BACKEND_URL}/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        top_k: limit || 50,
      }),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.error || 'Search failed')
    }

    const data = await response.json()

    const endTime = performance.now()
    const durationMs = Math.round(endTime - startTime)

    return NextResponse.json({
      success: true,
      results: data.results,
      count: data.results.points.length,
      query: data.query,
      durationMs,
    })
  } catch (error) {
    console.error('Search error:', error)
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}