'use client'

import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from '@/components/ui/input-group'
import { SearchIcon } from 'lucide-react'
import { useSearch } from '@/contexts/search-context'

export const SearchBar = () => {
  const { query, setQuery } = useSearch()

  return (
    <InputGroup>
      <InputGroupInput
        placeholder="Search for products"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <InputGroupAddon>
        <SearchIcon />
      </InputGroupAddon>
      <InputGroupAddon align="inline-end">
        <InputGroupButton>Search</InputGroupButton>
      </InputGroupAddon>
    </InputGroup>
  )
}
