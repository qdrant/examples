# Overview

### Quora Question Pairs

It is a large corpus of different questions and is used to detect similar/repeating questions by understanding the semantic meaning of them

### Qdrant

Qdrant is an Open-Source Vector Database and Vector Search Engine written in Rust. It provides fast and scalable vector similarity search service.

### Abstract

This notebook implements a search engine using the `Quora Duplicate Questions` dataset and the `Qdrant library`. It aims to identify similar questions based on user input queries.

### Methodology

Here's a detailed overview of implementation:

- Load the Quora Dataset and apply preprocessing steps.
- Vectorize the textual data and store in a vector space, where questions entered by users can be vectorized and compared in the same vector space - All these steps are covered by internal functionality of Qdrant.
- Several example queries are provided to demonstrate the functionality of the search engine.

### Summary

In summary, the notebook demonstrates how easily and efficiently, complete search engine can be created using Qdrant Vector Database and Client.

### Explore More!

- This notebook has been covered in an article on Medium: [Build a search engine in 5 minutes using Qdrant](https://medium.com/@raoarmaghanshakir040/build-a-search-engine-in-5-minutes-using-qdrant-f43df4fbe8d1)
- [E-Commerce Products Search Engine Using Qdrant](https://www.kaggle.com/code/sacrum/e-commerce-products-search-engine-using-qdrant)
- [Qdrant](https://qdrant.tech)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Qdrant Python Client Documentation](https://python-client.qdrant.tech)
- [Quora Question Pair](https://www.kaggle.com/competitions/quora-question-pairs)
