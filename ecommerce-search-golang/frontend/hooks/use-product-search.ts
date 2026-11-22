import { useQuery } from '@tanstack/react-query'
import { SearchParams, Product } from '@/app/api/search/route'

interface SearchResults {
  points: Product[]
}

interface SearchResponse {
  success: boolean
  results: SearchResults
  count: number
  query: string
  filters?: SearchParams['filters']
  error?: string
  durationMs: number
}

async function searchProducts(params: SearchParams): Promise<SearchResponse> {
  const response = await fetch('/api/search', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  })

  if (!response.ok) {
    throw new Error('Search request failed')
  }

  return response.json()
}

export function useProductSearch(
  params: SearchParams,
  enabled: boolean = true
) {
  return useQuery({
    queryKey: ['products', 'search', params.query, params.filters, params.limit],
    queryFn: () => searchProducts(params),
    enabled,
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}
