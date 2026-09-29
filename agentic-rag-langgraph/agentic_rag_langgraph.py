"""Agentic RAG with LangGraph and Qdrant.

Builds an agent that answers questions about the Hugging Face and
Transformers documentation. The agent chooses between two Qdrant
retrievers and a Brave web search tool, and loops until it has an answer.

Companion script for
https://qdrant.tech/documentation/tutorials-build-essentials/agentic-rag-langgraph/
"""

import os
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain_community.tools import BraveSearch
from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from qdrant_client import QdrantClient

# --- Configuration ---------------------------------------------------------

load_dotenv()
qdrant_key = os.getenv("QDRANT_KEY")
qdrant_url = os.getenv("QDRANT_URL")
brave_key = os.getenv("BRAVE_API_KEY")

if not os.getenv("OPENAI_API_KEY"):
    sys.exit("OPENAI_API_KEY is not set. Add it to your .env file.")
if not qdrant_url:
    sys.exit("QDRANT_URL is not set. Add it to your .env file.")

number_of_docs = 50

# --- Document processing ---------------------------------------------------


def preprocess_dataset(docs_list):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=700,
        chunk_overlap=50,
        disallowed_special=(),
    )
    return text_splitter.split_documents(docs_list)


hugging_face_doc = HuggingFaceDatasetLoader("m-ric/huggingface_doc", "text")
transformers_doc = HuggingFaceDatasetLoader("m-ric/transformers_documentation_en", "text")

hf_splits = preprocess_dataset(hugging_face_doc.load()[:number_of_docs])
transformer_splits = preprocess_dataset(transformers_doc.load()[:number_of_docs])

# --- Retrievers backed by Qdrant -------------------------------------------


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_key)


def create_retriever(collection_name, doc_splits):
    # Reuse the collection on later runs so documents are embedded only once
    if qdrant_client.collection_exists(collection_name):
        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            url=qdrant_url,
            api_key=qdrant_key,
            collection_name=collection_name,
        )
    else:
        vectorstore = QdrantVectorStore.from_documents(
            doc_splits,
            embeddings,
            url=qdrant_url,
            api_key=qdrant_key,
            collection_name=collection_name,
        )
    return vectorstore.as_retriever()


hf_retriever = create_retriever("hugging_face_documentation", hf_splits)
transformer_retriever = create_retriever("transformers_documentation", transformer_splits)

# --- Tools -----------------------------------------------------------------

hf_retriever_tool = create_retriever_tool(
    hf_retriever,
    "retriever_hugging_face_documentation",
    "Search and return information about hugging face documentation, it includes the guide and Python code.",
)

transformer_retriever_tool = create_retriever_tool(
    transformer_retriever,
    "retriever_transformer",
    "Search and return information specifically about transformers library",
)


@tool("web_search_tool")
def search_tool(query: str) -> str:
    """Search the web with Brave Search and return the top results."""
    search = BraveSearch.from_api_key(api_key=brave_key, search_kwargs={"count": 3})
    return search.run(query)


tools = [hf_retriever_tool, transformer_retriever_tool]
if brave_key:
    tools.append(search_tool)
else:
    print("BRAVE_API_KEY is not set; running without the web search tool.")

# --- Model and graph -------------------------------------------------------


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# ToolNode from langgraph.prebuilt executes every tool call in the last
# AI message and returns the results as ToolMessages.
tool_node = ToolNode(tools=tools)


def agent(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def route(state: State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"

    return END


graph_builder = StateGraph(State)

graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "agent",
    route,
    {"tools": "tools", END: END},
)

graph_builder.add_edge("tools", "agent")
graph_builder.add_edge(START, "agent")

graph = graph_builder.compile()

# --- Run -------------------------------------------------------------------


def run_agent(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "In the Transformers library, are there any multilingual models?"
    run_agent(question)
