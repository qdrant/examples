# -*- coding: utf-8 -*-
"""

# Overview

### Quora Question Pairs

It is a large corpus of different questions and is used to detect similar/repeating questions by understanding the semantic meaning of them

### Qdrant

Qdrant is an Open-Source Vector Database and Vector Search Engine written in Rust. It provides fast and scalable vector similarity search service.

### Abstract

This script implements a search engine using the `Quora Duplicate Questions` dataset and the `Qdrant library`. It aims to identify similar questions based on user input queries.

### Methodology

Here's a detailed overview of implementation:

- The script begins by loading the Quora dataset and extracting questions from it. Duplicate questions are removed to ensure uniqueness, and a sample of questions is taken to expedite processing. These questions are then indexed using the `Qdrant library`.
- A search function is defined to query the indexed questions for similar matches to the user input query. The top similar questions found are displayed as results.
- Several example queries are provided to demonstrate the functionality of the search engine. These queries cover various topics, allowing users to observe how the engine retrieves relevant matches based on semantic similarity.

### Summary
In summary, the script offers a practical demonstration of building a search engine for similar questions using real-world data and a specialized library. It provides a starting point for developing more sophisticated search functionalities and can be adapted for various applications requiring semantic similarity matching.

# Setting Up
1. Join the [Quora Question Pairs Competition on Kaggle](https://www.kaggle.com/competitions/quora-question-pairs).
2. Download the file [train.csv.zip](https://www.kaggle.com/competitions/quora-question-pairs/data?select=train.csv.zip).
3. Unzip the downloaded file.
4. Save the path to the dataset in `DATA_PATH`.
"""

DATA_PATH = "/kaggle/working/train.csv"

"""## Initialize Constants"""

# Name of Qdrant Collection for saving vectors
QD_COLLECTION_NAME = "collection_name"

# Sample size since the complete dataset is very long and can take long processing time
N = 30_000

"""# Dataset
- **Title:** Quora Question Pairs
- **Source:** Kaggle Competition
- **Link:** [Quora Question Pairs Competition on Kaggle](https://www.kaggle.com/competitions/quora-question-pairs)
"""

import pandas as pd

df = pd.read_csv(DATA_PATH)

print("Shape of DataFrame:", df.shape)
print("First 10 rows:")
df.head(10)

"""## Questions
Extracting Questions from dataset, removing duplications and sample a portion of data to use for search engine
"""

# extract the questions from df
questions = pd.concat([df['question1'], df['question2']], axis=0)

# remove all the duplicate questions
questions = questions.drop_duplicates()

# print total number of questions
print("Total Questions:", len(questions))

# sample questions from complete data to avoid long processing
questions = questions.sample(N)

# print first 10 questions
print("First 10 Questions:")
questions.iloc[:10]

"""# Qdrant"""

# !pip install qdrant-client[fastembed]

from qdrant_client import QdrantClient

client = QdrantClient(":memory:")

client.add(
    collection_name=QD_COLLECTION_NAME,
    documents=questions,
)

print("Completed")

def search(query):
    results = client.query(
        collection_name=QD_COLLECTION_NAME,
        query_text=query,
        limit=5
    )
    print("Query:", query)
    for i, result in enumerate(results):
        print()
        print(f"{i+1}) {result.document}")

search("what is the best earyly morning meal?")

search("How should one introduce themselves?")

search("Why is the Earth a sphere?")

"""# Explore More

- This notebook has been covered in an article on Medium: [Build a search engine in 5 minutes using Qdrant](https://medium.com/@raoarmaghanshakir040/build-a-search-engine-in-5-minutes-using-qdrant-f43df4fbe8d1)
- [E-Commerce Products Search Engine Using Qdrant](https://www.kaggle.com/code/sacrum/e-commerce-products-search-engine-using-qdrant)
- [Qdrant](https://qdrant.tech)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Qdrant Python Client Documentation](https://python-client.qdrant.tech)
- [Quora Question Pair](https://www.kaggle.com/competitions/quora-question-pairs)

"""

