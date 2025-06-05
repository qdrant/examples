# Multivector Representation Example

This example demonstrates how to use Qdrant's multi-vector search capabilities with both dense and late-interaction (ColBERT-style) embeddings for retrieval and reranking.

## Overview

- Connects to a Qdrant vector database instance.
- Loads two embedding models:
  - Dense embedding model (e.g., `BAAI/bge-small-en`)
  - Late interaction embedding model (e.g., `colbert-ir/colbertv2.0`)
- Indexes documents with both dense and ColBERT embeddings.
- Performs a search: first retrieves candidates with the dense vector, then reranks them using the ColBERT multivector.
- Returns the top reranked results.

## Requirements

- Python 3.8+
- Qdrant client
- fastembed

Install dependencies:
```bash
pip install qdrant-client fastembed
```

You also need a running Qdrant instance (default: `localhost:6333`).

## Usage

1. **Prepare your Qdrant server:**  
   Make sure Qdrant is running locally on port 6333.

2. **Run the example:**  
   ```bash
   python multivector.py
   ```

3. **Expected Output:**  
   The script will print the top reranked results for the query:
   ```
   How does AI help in medicine?
   ```

## How it works

1. **Connect to Qdrant:**
   ```python
   client = QdrantClient(host="localhost", port=6333)
   ```

2. **Load Embedding Models:**
   - Dense: `TextEmbedding("BAAI/bge-small-en")`
   - ColBERT: `LateInteractionTextEmbedding("colbert-ir/colbertv2.0")`

3. **Generate Embeddings for Documents and Query.**

4. **Create Qdrant Collection:**  
   With both dense and ColBERT vector configurations.

5. **Upsert Documents:**  
   Each document is stored with both types of vectors and its text payload.

6. **Search:**  
   - Prefetch using dense vector (fast, broad retrieval).
   - Rerank using ColBERT vector (precise, late interaction).

7. **Display Results:**  
   Prints the top 3 reranked documents.

## Example Output

```python
points=[
  ScoredPoint(id=1, ..., payload={'text': 'Self-driving cars use AI to detect obstacles and make driving decisions.'}),
  ScoredPoint(id=2, ..., payload={'text': 'AI is transforming customer service through chatbots and automation.'}),
  ScoredPoint(id=0, ..., payload={'text': 'Artificial intelligence is used in hospitals for cancer diagnosis and treatment.'})
]
```

## Customization

- Change the `documents` and `query_text` variables to use your own data.
- Adjust collection and vector parameters as needed for your use case.

## References

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [fastembed Documentation](https://github.com/qdrant/fastembed)
