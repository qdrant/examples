# Find Products Nearby With Qdrant Score Boosting

Build e-commerce search with real [Wayfair WANDS](https://github.com/wayfair/WANDS) product text, stock and radius filters, a free-returns bonus, and geographic decay. Inspect each score contribution and use WANDS relevance labels to check what the ranking policy displaces.

[Open the notebook](./ecommerce-search.ipynb) and run its cells in order in a standard CPU runtime. Allow about 50 minutes including experiments; this is an estimate pending a learner walkthrough. First-run downloads depend on your connection. No Qdrant account, API key, GPU, paid inference, or Docker is required. Google Colab requires Google sign-in.

The default query is WANDS query 418, `writing desk`. The notebook indexes all **1,123 unique products judged for that query**. It removes one repeated, agreeing judgment before selection. Product names, descriptions, query text, and relevance labels are real WANDS data. **Pickup coordinates, stock, and free-return flags are deterministic simulations**, not Wayfair locations, inventory, or policies. Each product has one simulated offer.

WANDS contains 42,994 products, 480 queries, and 233,448 judgment rows. The notebook downloads its three tab-separated files (about 97 MB) from commit `3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5` and checks their SHA-256 digests. The source dataset is provided by Wayfair under the [MIT license](https://github.com/wayfair/WANDS/blob/3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5/LICENSE).

The notebook demonstrates:

- Semantic retrieval using product names and descriptions, with locally generated embeddings.
- Hard eligibility with simulated stock and a 15 km pickup radius.
- Boolean free-return bonuses and smooth Gaussian distance bonuses.
- A table reconciling actual Qdrant scores with each contribution.
- Excessive proximity weighting and a narrow prefetch that omits useful candidates.
- NDCG@5 over the same eligible judged pool for each policy, with the individual labels shown.
- Changing the shopper's location and explicitly tuning weights.
- Checks for zero-weight equivalence, missing coordinates, eligibility, empty results, and score arithmetic.

The relevance audit is a **single-query, within-pool diagnostic**. It is not a full-catalog or held-out benchmark. WANDS labels evaluate product/query relevance, not the simulated operating preferences. The notebook does not claim improved conversion, shopper satisfaction, or production search quality.

The runtime uses Qdrant **1.20.2**, `qdrant-client` **1.20.0**, FastEmbed **0.8.0**, and BGE small English v1.5 at a pinned ONNX revision. Embedding batches of 32 limit peak memory use. It downloads a checksum-verified real Qdrant server and starts it on an isolated localhost port. Linux x86-64, macOS Apple Silicon, and macOS Intel bootstraps are included. Set `CLEAN_UP=True` in the final cell to stop that server and remove its temporary data.

The optional extension discusses the larger [Amazon ESCI embeddings bucket](https://huggingface.co/buckets/itsclelia/esci-products-embeddings). Download it separately with `hf buckets sync hf://buckets/itsclelia/esci-products-embeddings ./storage` after checking its current size and available disk space. This extension is not downloaded or executed by the notebook.

The bucket README inspected on November 25, 2026 describes 5,781,746 products and several vector representations with different coverage. Its dense model is not identified in the inspected README; 384 dimensions alone do not imply compatibility with this notebook's BGE model. Its 2,000 held-out query titles are product titles, not verified relevance judgments. Check current metadata, model identities, product-ID joins, and vector configuration before adapting retrieval. Geo and stock data still need a separate operational source.

References: [WANDS](https://github.com/wayfair/WANDS), [Qdrant score boosting](https://qdrant.tech/documentation/search/search-relevance/), [Qdrant v1.20.2](https://github.com/qdrant/qdrant/releases/tag/v1.20.2), and [the pinned embedding model](https://huggingface.co/qdrant/bge-small-en-v1.5-onnx-q/tree/aa8f8b060edb00e03bfdd08813a2949946c8ba55).
