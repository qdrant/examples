import path from 'path'
import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  output: 'standalone',
  // Ensure these packages are not bundled on the server (works with Turbopack)
  // outputFileTracingRoot: path.join(__dirname, '../'),
  serverExternalPackages: [
    'onnxruntime-node',
    'fastembed',
    '@anush008/tokenizers',
  ],
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname:
          'qdrant-nextjs-demo-product-images.s3.us-east-1.amazonaws.com',
      },
    ],
  },
}

export default nextConfig
