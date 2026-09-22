import json
from pathlib import Path

from qdrant_client import QdrantClient, models

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION = "items"
QUERIES = [
    "spicy vegetarian soup",
    "fresh seafood on a bun",
    "quick healthy lunch bowl",
]


def print_table(headers, rows):
    widths = [
        max(len(str(row[i])) for row in ([headers] + rows))
        for i in range(len(headers))
    ]

    def line(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    print(line(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(line(row))
    print()


items = json.loads(Path(__file__).with_name("menu-items.json").read_text())

print(f"Menu dataset: {len(items)} items\n")
print_table(
    ["Name", "Category", "Price"],
    [[item["name"], item["category"], item["price"]] for item in items],
)

client = QdrantClient(url="http://localhost:6333")

if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)

client.upsert(
    collection_name=COLLECTION,
    points=[
        models.PointStruct(
            id=i,
            vector=models.Document(
                text=f"{item['name']}. {item['description']}", model=EMBED_MODEL
            ),
            payload=item,
        )
        for i, item in enumerate(items)
    ],
)

print(f"Upserted {len(items)} items into the '{COLLECTION}' collection.\n")

for query in QUERIES:
    print(f'Query: "{query}"')
    print(
        "  client.query_points(\n"
        f'      collection_name="{COLLECTION}",\n'
        f'      query=models.Document(text="{query}", model="{EMBED_MODEL}"),\n'
        "      limit=3,\n"
        "  )\n"
    )
    results = client.query_points(
        collection_name=COLLECTION,
        query=models.Document(text=query, model=EMBED_MODEL),
        limit=3,
    ).points
    print_table(
        ["Name", "Category", "Price", "Score"],
        [
            [r.payload["name"], r.payload["category"], r.payload["price"], f"{r.score:.4f}"]
            for r in results
        ],
    )
