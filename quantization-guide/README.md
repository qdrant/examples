# Vector Quantization in Qdrant: Reproducible Measurements

Companion notebook for the article [Vector Quantization in Qdrant: Which Method to Use and What You Trade](https://qdrant.tech/articles/vector-quantization-guide/).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qdrant/examples/blob/master/quantization-guide/quantization_guide.ipynb)

It downloads SciFact with its published Cohere embed-english-v3.0 vectors, computes exact neighbors with NumPy, builds one Qdrant collection per quantization method, and measures recall@10 with rescoring off and on, a tuning sweep against a recall target, the `turbo4` storage datatype, and the sign-imbalance statistic for binary quantization.

Requirements: a Qdrant server at version 1.19 or later (a free Qdrant Cloud cluster or `docker run -p 6333:6333 qdrant/qdrant:v1.19.2`). No embedding API key is needed.

The full benchmark behind the article, with four datasets, two models, memory, throughput, and cold-start experiments, lives in [qdrant-labs/quantization-comparison-bench](https://github.com/qdrant-labs/quantization-comparison-bench).
