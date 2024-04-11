---
title: RAG with Langchain and Qdrant on Scaleway
weight: 22
---

| Time: 90 min | Level: Advanced |  |    |
|--------------|-----------------|--|----|
### Installation of Required Libraries

## Langchain x Qdrant: RAG Demo with Web Scraping

This section introduces the demonstration of building a Retrieval-Augmented Generation (RAG) model that combines web scraping with the capabilities of Langchain and Qdrant. The RAG model enhances the generation of answers by first retrieving relevant documents. Qdrant serves as the vector search engine for retrieval, while GPT-3.5, developed by OpenAI, is utilized as the generator for producing answers. This setup showcases the integration of advanced search and AI language processing to improve information retrieval and generation tasks.

## Setting Up

To prepare the environment for working with Qdrant and related libraries, it's necessary to install all required Python packages. This can be done using Poetry, a tool for dependency management and packaging in Python. The code snippet imports various libraries essential for the tasks ahead, including `bs4` for parsing HTML and XML documents, `langchain` and its community extensions for working with language models and document loaders, and `Qdrant` for vector storage and retrieval. These imports lay the groundwork for utilizing Qdrant alongside other tools for natural language processing and machine learning tasks.

```python
import getpass
import os

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

### Setting Up the OpenAI API Key

```python
os.environ["OPENAI_API_KEY"] = getpass.getpass()
```

### Initializing the Language Model

```python
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
```

It is here that we configure both the Embeddings and LLM. You can replace this with your own models using Ollama or other services. Scaleway has some great [GPU Instances](https://www.scaleway.com/en/gpu-instances/) too - including H100 on the higher end, and soon L4 for everything small.

## Download and Index

To begin working with blog post contents, the process involves loading and parsing the HTML content. This is achieved using `urllib` and `BeautifulSoup`, which are tools designed for such tasks. After the content is loaded and parsed, it is indexed using Qdrant, a powerful tool for managing and querying vector data. The code snippet demonstrates how to load, chunk, and index the contents of a blog post by specifying the URL of the blog and the specific HTML elements to parse. This step is crucial for preparing the data for further processing and analysis with Qdrant.

```python
# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

```

### Chunking before Indexing

When dealing with large documents, such as a blog post exceeding 42,000 characters, it's crucial to manage the data efficiently for processing. Many models have a limited context window and struggle with long inputs, making it difficult to extract or find relevant information. To overcome this, the document is divided into smaller chunks. This approach enhances the model's ability to process and retrieve the most pertinent sections of the document effectively.

In this scenario, the document is split into chunks using the `RecursiveCharacterTextSplitter` with a specified chunk size and overlap. This method ensures that no critical information is lost between chunks. Following the splitting, these chunks are then indexed into Qdrant—a vector database for efficient similarity search and storage of embeddings. The `Qdrant.from_documents` function is utilized for indexing, with documents being the split chunks and embeddings generated through `OpenAIEmbeddings`. The entire process is facilitated within an in-memory database, signifying that the operations are performed without the need for persistent storage, and the collection is named "lilianweng" for reference.

This chunking and indexing strategy significantly improves the management and retrieval of information from large documents, making it a practical solution for handling extensive texts in data processing workflows.

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Qdrant.from_documents(
    documents=splits, embedding=OpenAIEmbeddings(), location=":memory:", collection_name="lilianweng"
)
```

### Retrieve and Generate

In this section, the process of retrieving information and generating content using a vector store and a language model is outlined. The `vectorstore` is utilized as a retriever to fetch relevant documents based on vector similarity. The `hub.pull("rlm/rag-prompt")` function is used to pull a specific prompt from a repository, which is designed to work with retrieved documents and a question to generate a response.

The `format_docs` function formats the retrieved documents into a single string, preparing them for further processing. This formatted string, along with a question, is passed through a chain of operations. Firstly, the context (formatted documents) and the question are processed by the retriever and the prompt. Then, the result is fed into a large language model (`llm`) for content generation. Finally, the output is parsed into a string format using `StrOutputParser()`.

This chain of operations demonstrates a sophisticated approach to information retrieval and content generation, leveraging both the semantic understanding capabilities of vector search and the generative prowess of large language models.

```python
# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### Invoking the RAG Chain

```python
rag_chain.invoke("What is Task Decomposition?")
```

## Deploying Langchain Applications on Scaleway
Scaleway has serverless [Functions](https://www.scaleway.com/en/serverless-functions/) and serverless [Jobs](https://www.scaleway.com/en/serverless-jobs/) -- ideal for embedding creation when doing a bulk operation.

Their French deployment regions e.g. France are excellent for network latency and data sovereignty. Need a GPU? [Render with P100](https://www.scaleway.com/en/gpu-render-instances/) is there for you.