import os


collection_name = "mscoco"
qdrant_api_key = os.environ['QDRANT_API_KEY']
qdrant_url = os.environ['QDRANT_URL']

examples = [["a woman holding a smartphone", "images"],
            ["a man riding a bicycle", "images"], ["a view of a city at night", "images"]]

description = """
this is a semantic image search demo with the [MSCOCO](https://cocodataset.org)
dataset backed by [Qdrant Cloud](https://qdrant.to/cloud).
It encodes the textual query to a 512-dimensional vector with CLIP and searches for images matching the query with image or caption embeddings
stored in a free-tier Qdrant Cloud cluster.
"""

article = """
### About Qdrant

<p align="center">
  <img height="100" src="https://github.com/qdrant/qdrant/raw/master/docs/logo.svg" alt="Qdrant">
</p>

<p align="center">
    <b>Vector Search Engine for the next generation of AI applications</b>
</p>

Qdrant (read: quadrant ) is a vector similarity search engine and vector database. It provides a production-ready service with a convenient API to store, search, and manage points - vectors with an additional payload. Qdrant is tailored to extended filtering support. It makes it useful for all sorts of neural-network or semantic-based matching, faceted search, and other applications.

Qdrant is written in Rust 🦀, which makes it fast and reliable even under high load.

With Qdrant, embeddings or neural network encoders can be turned into full-fledged applications for matching, searching, recommending, and much more!

Also available as managed solution in the [Qdrant Cloud](https://qdrant.to/cloud).

With the generous free tier provided by Qdrant Cloud and Huggingface Spaces, you can set your vector search applications up and running in no time. Please remember to safely store your credentials as secrets in HF Spaces, just as is done in this demo.

### Find us

<p align=center>
<a href="https://qdrant.to/discord"><img src="https://img.shields.io/badge/Discord-Qdrant-5865F2.svg?logo=discord" alt="Discord"></a>
</p>
"""
