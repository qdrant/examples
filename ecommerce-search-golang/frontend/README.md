# Qdrant Next.js E-commerce Demo

A **completely Python-free (almost)** vector search demo for e-commerce, built to showcase local-first semantic search with modern web technologies.

## Stack

- **Next.js** + **shadcn/ui** for the frontend
- **fastembed-js** for client-side embeddings
- **Qdrant** (local instance) for vector storage and search
- **100K e-commerce products** embedded with:
  - SPLADEv1 (sparse vectors)
  - bge-small-en-v1.5 (dense vectors)
  - Link to dataset: [hm_ecommerce_products_100k](https://huggingface.co/datasets/nleroy917/hm_ecommerce_products)

## Getting Started

1. **Install dependencies**

   ```bash
   npm install
   ```

2. **Start Qdrant locally**

   ```bash
   docker run -p 6333:6333 qdrant/qdrant:latest
   ```

3. **Run the development server**

   ```bash
   npm run dev
   ```

4. **Open the app**

   Navigate to [http://localhost:3000](http://localhost:3000)

## Why This Demo?

This project demonstrates that you can build sophisticated vector search experiences entirely in JavaScript/TypeScript—no Python required. Everything from client-side embedding generation to semantic search runs in the browser and Node.js ecosystem.
