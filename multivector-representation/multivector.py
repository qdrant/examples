from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, LateInteractionTextEmbedding

# 1. Connect to Qdrant server
client = QdrantClient(host="localhost", port=6333)

# 2. Load embedding models
dense_model = TextEmbedding("BAAI/bge-small-en")  # Any dense model
colbert_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")  # Late interaction

# 3. Example documents and query
documents = [
    "Artificial intelligence is used in hospitals for cancer diagnosis and treatment.",
    "Self-driving cars use AI to detect obstacles and make driving decisions.",
    "AI is transforming customer service through chatbots and automation."
]
query_text = "How does AI help in medicine?"

# 4. Generate embeddings
dense_doc_vectors = list(dense_model.embed(documents))
dense_query_vector = list(dense_model.embed([query_text]))[0]

colbert_doc_vectors = list(colbert_model.embed(documents))
colbert_query_vector = list(colbert_model.embed([query_text]))[0]

# 5. Create collection with dense + multivector configuration
collection_name = "dense_multivector_search"
client.create_collection(
    collection_name=collection_name,
    vectors_config={
        "dense": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
            # Leave HNSW indexing ON for dense
        ),
        "colbert": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
            hnsw_config=models.HnswConfigDiff(m=0)  #  Disable HNSW for reranking
        )
    }
)

# 6. Upload documents with both dense and multivector embeddings
points = [
    models.PointStruct(
        id=i,
        vector={
            "dense": dense_doc_vectors[i],
            "colbert": colbert_doc_vectors[i]
        },
        payload={"text": documents[i]}
    )
    for i in range(len(documents))
]
client.upsert(collection_name=collection_name, points=points)

# 7. Search using dense vector (prefetch), then rerank with multivector in one query
results = client.query_points(
    collection_name=collection_name,
    prefetch=models.Prefetch(
        query=dense_query_vector, 
        using="dense",
        limit=100
    ),
    query=colbert_query_vector,
    using="colbert",
    limit=3,
    with_payload=True
)

# 8. Display final reranked results
print(results)
"""
points=[ScoredPoint(id=1, version=0, score=5.0332537, payload={'text': 'Self-driving cars use AI to detect obstacles and make driving decisions.'}, vector=None, shard_key=None, order_value=None), ScoredPoint(id=2, version=0, score=4.7417374, payload={'text': 'AI is transforming customer service through chatbots and automation.'}, vector=None, shard_key=None, order_value=None), ScoredPoint(id=0, version=0, score=4.727172, payload={'text': 'Artificial intelligence is used in hospitals for cancer diagnosis and treatment.'}, vector=None, shard_key=None, order_value=None)]
"""