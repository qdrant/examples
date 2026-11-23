# Find Film Locations With Qdrant Score Boosting

Build a location-scouting search that combines semantic similarity with filming permission, a distance limit, on-site power, and a geographic proximity bonus.

[Open the notebook](./film-location-search.ipynb) and run its cells in order. In Google Colab, use a standard CPU runtime. Allow roughly 35–50 minutes including the experiments; first-run downloads depend on your connection.

The notebook is self-contained. It includes 40 fictional Warsaw-area location records, downloads a checksum-verified Qdrant 1.20.2 server, and generates real embeddings with FastEmbed 0.8.0 and a pinned revision of BGE small English v1.5. It requires no Qdrant account, API key, paid inference, GPU, or Docker. Google Colab itself requires Google sign-in. Python 3.12 is the validation environment.

The records are teaching fixtures, not real venue listings or availability claims. Their purpose is to make ranking changes inspectable, not to measure production search quality.

The notebook demonstrates:

- Semantic retrieval before and after permission and radius filters.
- Boolean facility bonuses, Gaussian geographic decay, and their combined formula.
- A table reconciling every final score with its individual contributions.
- An excessive proximity weight and a candidate pool that misses useful results.
- Changing the crew base without re-embedding the scene description.
- Checks for eligibility, missing coordinates, empty results, and zero-weight equivalence.

The bootstrap supports Linux x86-64, macOS Apple Silicon, and macOS Intel. Each run starts its own server on a free localhost port and uses a separate temporary directory. Set `CLEAN_UP=True` in the final cell when finished to stop that server and remove its temporary files.

References: [Qdrant score boosting](https://qdrant.tech/documentation/search/search-relevance/), [Qdrant v1.20.2](https://github.com/qdrant/qdrant/releases/tag/v1.20.2), and [the pinned embedding model](https://huggingface.co/qdrant/bge-small-en-v1.5-onnx-q/tree/aa8f8b060edb00e03bfdd08813a2949946c8ba55).
