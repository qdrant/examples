from qdrant_client import QdrantClient
import openai
import os
from qdrant_client.http import models as rest
from openai import OpenAI

qdrant_client = QdrantClient(url=os.getenv("QDRANT_HOST"))

def search_similar_transcripts(query_text, user, top_k=10):
    # 1️⃣ Generate the embedding for your query
    openai_client = OpenAI(api_key=user.openai_api_key_decrypted)
    emb_response = openai_client.embeddings.create(
        input=[query_text],
        model="text-embedding-ada-002"
    )
    query_embedding = emb_response.data[0].embedding

    # 2️⃣ Search Qdrant using that vector
    hits = qdrant_client.search(
        collection_name="video_transcripts",
        query_vector=query_embedding,
        limit=top_k,
        query_filter=rest.Filter(
            must=[
                rest.FieldCondition(
                    key="user_id",
                    match=rest.MatchValue(value=str(user.id))
                )
            ]
        )
    )

    # 3️⃣ Only keep hits whose payload actually has a non‐empty "transcript"
    return [hit.payload for hit in hits if hit.payload.get("transcript")]


def generate_video_idea(user,query_text, similar_payloads):
    existing_transcripts = "\n---\n".join(
    p["transcript"][:1000] for p in similar_payloads
)
    prompt = f"""
You are a world-class YouTube strategist.

Your job is to propose a brand-new video idea that is:
- Similar to existing transcript topics.
- Visually engaging.
- Potentially viral.
- Aligned with the query: "{query_text}"

Here are related past transcripts:
{existing_transcripts}

Return your idea in 1 short paragraph, with a unique title suggestion in quotes.
"""



    response = openai.chat.completions.create(
        model=user.openai_model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
