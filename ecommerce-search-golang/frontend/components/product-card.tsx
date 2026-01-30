import Image from 'next/image'

interface ProductPayload {
  article_id: string
  product_code: string
  prod_name: string
  product_type_name: string
  product_group_name: string
  graphical_appearance_name: string
  colour_group_name: string
  perceived_colour_value_name: string
  perceived_colour_master_name: string
  department_name: string
  index_name: string
  index_group_name: string
  section_name: string
  garment_group_name: string
  detail_desc: string
  image_url: string
}

interface ProductCardProps {
  product: ProductPayload
}

export function ProductCard({ product }: ProductCardProps) {
  return (
    <div className="border rounded-lg p-4 space-y-3 hover:shadow-lg transition-shadow">
      <div className="relative w-full aspect-square bg-gray-100 rounded-md overflow-hidden">
        {product.image_url ? (
          <Image
            src={product.image_url}
            alt={product.prod_name}
            fill
            className="object-cover"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-400">
            No Image
          </div>
        )}
      </div>
      <div className="space-y-1">
        <h3 className="font-semibold text-lg line-clamp-2">
          {product.prod_name}
        </h3>
        <p className="text-sm text-muted-foreground">
          {product.product_type_name}
        </p>
        <div className="flex flex-wrap gap-2 text-sm text-muted-foreground">
          {product.index_name && (
            <span className="bg-accent px-2 py-1 rounded">
              {product.index_name}
            </span>
          )}
          {product.colour_group_name && (
            <span className="bg-accent px-2 py-1 rounded">
              {product.colour_group_name}
            </span>
          )}
          {product.graphical_appearance_name && (
            <span className="bg-accent px-2 py-1 rounded">
              {product.graphical_appearance_name}
            </span>
          )}
        </div>
        {product.detail_desc && (
          <p className="text-sm text-muted-foreground line-clamp-2">
            {product.detail_desc}
          </p>
        )}
      </div>
    </div>
  )
}

export type { ProductPayload }
