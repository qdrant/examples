# Agentic RAG with LangGraph and Qdrant

Companion code for the tutorial [Agentic RAG with LangGraph](https://qdrant.tech/documentation/tutorials-build-essentials/agentic-rag-langgraph/).

The script builds a LangGraph agent that answers questions about the Hugging Face and Transformers documentation. It loads two datasets from the Hugging Face Hub, splits and embeds them with OpenAI `text-embedding-3-small`, stores them in two Qdrant collections, and gives a `gpt-4o` agent three tools: a retriever for each collection and a Brave web search tool.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Embeddings and the agent model |
| `QDRANT_URL` | Your Qdrant Cloud cluster URL, or `http://localhost:6333` for a local instance |
| `QDRANT_KEY` | Qdrant API key, leave empty for a local instance without auth |
| `BRAVE_API_KEY` | Optional. Enables the web search tool |

## Run

```bash
python agentic_rag_langgraph.py "In the Transformers library, are there any multilingual models?"
```

The first run downloads the datasets and embeds the first 50 documents of each, which takes a minute or two. Later runs reuse the Qdrant collections.
