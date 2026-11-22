'use client'

import { createContext, useContext, useState, ReactNode } from 'react'

export type ColorFilter = 'black' | 'white' | 'grey' | 'red' | 'orange' | 'yellow' | 'green' | 'blue' | 'turquoise' | 'purple' | 'pink' | 'beige' | 'brown' | 'gold' | 'silver' | 'bronze'
export type GenderFilter = 'all' | 'divided' | 'ladieswear' | 'menswear' | 'baby-children' | 'sport'
export type PriceRange = [number, number]

export interface Filters {
  color?: ColorFilter
  gender: GenderFilter
  price: PriceRange
}

interface SearchContextType {
  query: string
  setQuery: (query: string) => void
  filters: Filters
  setFilters: (filters: Filters) => void
  updateFilter: <K extends keyof Filters>(key: K, value: Filters[K]) => void
  resetFilters: () => void
}

const SearchContext = createContext<SearchContextType | undefined>(undefined)

export function useSearch() {
  const context = useContext(SearchContext)
  if (context === undefined) {
    throw new Error('useSearch must be used within a SearchProvider')
  }
  return context
}

interface SearchProviderProps {
  children: ReactNode
}

const DEFAULT_FILTERS: Filters = {
  gender: 'all',
  price: [0, 2500],
}

export function SearchProvider({ children }: SearchProviderProps) {
  const [query, setQuery] = useState('')
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS)

  const updateFilter = <K extends keyof Filters>(key: K, value: Filters[K]) => {
    setFilters((prev) => ({ ...prev, [key]: value }))
  }

  const resetFilters = () => {
    setFilters(DEFAULT_FILTERS)
    setQuery('')
  }

  const value = {
    query,
    setQuery,
    filters,
    setFilters,
    updateFilter,
    resetFilters,
  }

  return (
    <SearchContext.Provider value={value}>{children}</SearchContext.Provider>
  )
}
