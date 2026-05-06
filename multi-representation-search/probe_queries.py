"""Try candidate queries against the multi-representation collection and print
all 5 step outputs side-by-side so we can pick the one with the best narrative arc."""

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

COLLECTION = "arxiv_multi_repr"
client = QdrantClient("http://localhost:6333")
dense_model = TextEmbedding("BAAI/bge-small-en-v1.5")
sparse_model = SparseTextEmbedding("Qdrant/bm25")


def to_sparse(s):
    return models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())


def embed(query):
    d = next(iter(dense_model.query_embed([query]))).tolist()
    s = to_sparse(next(iter(sparse_model.query_embed([query]))))
    return d, s


def step1(q, k=5):
    d, _ = embed(q)
    return client.query_points(COLLECTION, query=d, using="dense_chunk", limit=k).points


def step2(q, k=5):
    d, s = embed(q)
    return client.query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=d, using="dense_chunk", limit=50),
            models.Prefetch(query=s, using="sparse_keywords", limit=50),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
    ).points


def step3(q, k=5):
    d, s = embed(q)
    return client.query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=d, using="dense_chunk", limit=50),
            models.Prefetch(query=d, using="dense_title", limit=50),
            models.Prefetch(query=s, using="sparse_keywords", limit=50),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
    ).points


def step4(q, k=5):
    d, s = embed(q)
    return client.query_points_groups(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=d, using="dense_chunk", limit=100),
            models.Prefetch(query=d, using="dense_title", limit=100),
            models.Prefetch(query=s, using="sparse_keywords", limit=100),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        group_by="document_id",
        group_size=3,
        limit=k,
    ).groups


def step5(q, k=5):
    d, s = embed(q)
    return client.query_points_groups(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=d, using="dense_chunk", limit=100),
            models.Prefetch(query=d, using="dense_title", limit=100),
            models.Prefetch(query=s, using="sparse_keywords", limit=100),
        ],
        query=models.FormulaQuery(
            formula=models.SumExpression(sum=[
                "$score[0]",
                models.MultExpression(mult=[0.5, "$score[1]"]),
                models.MultExpression(mult=[0.3, "$score[2]"]),
            ]),
            defaults={"$score[1]": 0.0, "$score[2]": 0.0},
        ),
        group_by="document_id",
        group_size=3,
        limit=k,
    ).groups


def title_of(item):
    p = item.hits[0] if hasattr(item, "hits") else item
    return p.payload["title"].replace("\n", " ").strip()


def doc_id_of(item):
    p = item.hits[0] if hasattr(item, "hits") else item
    return p.payload["document_id"]


def run_query(q):
    print(f"\n{'=' * 100}\nQUERY: {q!r}\n{'=' * 100}")
    for name, fn in [("step1 dense-only", step1),
                     ("step2 +sparse RRF", step2),
                     ("step3 +title RRF", step3),
                     ("step4 grouped   ", step4),
                     ("step5 formula   ", step5)]:
        results = fn(q, k=5)
        seen = set()
        dup_count = 0
        lines = []
        for i, item in enumerate(results, 1):
            doc = doc_id_of(item)
            t = title_of(item)
            dup = " [DUP]" if doc in seen else ""
            if doc in seen:
                dup_count += 1
            seen.add(doc)
            lines.append(f"    {i}. {t[:90]}{dup}")
        print(f"\n  [{name}]  unique_docs={len(seen)}  dup_chunks={dup_count}")
        for ln in lines:
            print(ln)


CANDIDATES = [
    "diffusion models for image synthesis",  # current
    "transformer architecture for language modeling",
    "adversarial examples in deep learning",
    "contrastive self-supervised representation learning",
    "graph neural networks for node classification",
    "reinforcement learning from human feedback",
    "neural machine translation with attention",
    "knowledge distillation for model compression",
    "object detection in autonomous driving",
    "few-shot learning with meta-learning",
    "BERT pretraining for sentence classification",
    "convolutional neural network for image classification",
    "generative adversarial networks for image generation",
    "variational autoencoder for representation learning",
    "speech recognition with recurrent neural networks",
]

if __name__ == "__main__":
    import sys
    queries = sys.argv[1:] if len(sys.argv) > 1 else CANDIDATES
    for q in queries:
        run_query(q)
