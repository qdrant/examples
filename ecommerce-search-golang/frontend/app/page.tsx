'use client'

import { SearchBar } from '@/components/search-bar'
import { ProductCard } from '@/components/product-card'
import { ProductGridSkeleton } from '@/components/product-card-skeleton'
import { useDebounce } from '@uidotdev/usehooks'
import { useSearch } from '@/contexts/search-context'
import { useProductSearch } from '@/hooks/use-product-search'

export default function Home() {
  const { query } = useSearch()
  const debouncedQuery = useDebounce(query, 300)
  const { data, isLoading, isPending } = useProductSearch(
    {
      query: debouncedQuery,
    },
    !!debouncedQuery
  )

  const showLoading = (isLoading || isPending) && !!debouncedQuery

  return (
    <div className="flex flex-col gap-6 p-4">
      <div className="flex flex-row items-center max-w-225">
        <SearchBar />
      </div>

      {showLoading && (
        <div className="w-full">
          <ProductGridSkeleton />
        </div>
      )}

      {!showLoading && data?.results && data.results.points.length > 0 && (
        <div className="w-full">
          <div className="mb-4">
            <p className="text-sm text-muted-foreground">
              Found {data.count} {data.count === 1 ? 'result' : 'results'} for
              &quot;{data.query}&quot;
              {data.durationMs !== undefined && ` in ${data.durationMs}ms`}
            </p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {data.results.points.map((product) => (
              // @ts-expect-error its ok...
              <ProductCard key={product.id} product={product.payload} />
            ))}
          </div>
        </div>
      )}

      {!showLoading && data?.results && data.results.points.length === 0 && (
        <div className="w-full text-center py-12">
          <p className="text-muted-foreground">
            No results found for &quot;{data.query}&quot;
          </p>
        </div>
      )}

      {!showLoading && !debouncedQuery && (
        <div className="w-full text-center py-12">
          <p className="text-muted-foreground">
            Start typing to search for products
          </p>
        </div>
      )}
    </div>
  )
}
