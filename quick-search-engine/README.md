# Qdrant Search Engine

This code contains code and instructions for building a search engine using Qdrant. Qdrant is particularly well-suited for tasks involving similarity search, such as recommendation systems, image or text retrieval, and clustering.

## Article Overview

The associated Medium article titled ["Build a search engine in 5 minutes using Qdrant"](https://medium.com/@armaghanshakir/build-a-search-engine-in-5-minutes-using-qdrant-5c7a3ef6ac5) provides a step-by-step guide on building a search engine using Qdrant.

## Instructions

### Setting Up

1. Join the [Quora Question Pairs Competition on Kaggle](https://www.kaggle.com/c/quora-question-pairs).
2. Download the file `train.csv.zip`.
3. Unzip the downloaded file.
4. Save the path to the dataset in the `DATA_PATH` variable.

```python
DATA_PATH = "/kaggle/working/train.csv" # path to your train.csv file
```

### Initializing Constants

```python
# Name of Qdrant Collection for saving vectors
QD_COLLECTION_NAME = "collection_name"

# Sample size since the complete dataset is very long and can take long processing time
N = 30_000
```

### Loading Dataset

Load the dataset using pandas in Python:

```python
import pandas as pd

df = pd.read_csv(DATA_PATH)

print("Shape of DataFrame:", df.shape)
print("First 10 rows:")
df.head(10)
```

### Extracting Questions

Extract questions from the dataframe, remove duplicates, and create a sample to see results on a part of the data:

```python
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
```

### Installing Qdrant and Adding Documents

First, install qdrant with fastembed:

```bash
!pip install qdrant-client[fastembed]
```

Then add the documents to the Qdrant vector space:

```python
from qdrant_client import QdrantClient

client = QdrantClient(":memory:")

client.add(
    collection_name=QD_COLLECTION_NAME,
    documents=questions,
)

print("Completed")
```

### Creating a Search Function

Create a search function that processes the query and prints the results:

```python
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

search("what is the best early morning meal?")
search("How should one introduce themselves?")
search("Why is the Earth a sphere?")
```

## Additional Resources

- [E-Commerce Products Search Engine Using Qdrant](https://medium.com/@armaghanshakir/e-commerce-products-search-engine-using-qdrant-b65dc6ab1983)
- [Qdrant Documentation](https://qdrant.github.io/)
- [Qdrant Python Client Documentation](https://qdrant-client.readthedocs.io/en/latest/)