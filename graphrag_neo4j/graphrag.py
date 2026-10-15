from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
import uuid
import os

# Load environment variables
load_dotenv()

# Get credentials from environment variables
qdrant_key = os.getenv("QDRANT_KEY")
qdrant_url = os.getenv("QDRANT_URL")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
openai_key = os.getenv("OPENAI_API_KEY")

# Initialize Neo4j driver
neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_key
)

class single(BaseModel):
    node: str
    target_node: str
    relationship: str

class GraphComponents(BaseModel):
    graph: list[single]

client = OpenAI()

def openai_llm_parser(prompt):
    completion = client.chat.completions.create(
        model="gpt-5-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": 
                   
                """ You are a precise graph relationship extractor. Extract all 
                    relationships from the text and format them as a JSON object 
                    with this exact structure:
                    {
                        "graph": [
                            {"node": "Person/Entity", 
                             "target_node": "Related Entity", 
                             "relationship": "Type of Relationship"},
                            ...more relationships...
                        ]
                    }
                    Include ALL relationships mentioned in the text, including 
                    implicit ones. Be thorough and precise. """
                    
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    return GraphComponents.model_validate_json(completion.choices[0].message.content)
    
def extract_graph_components(raw_data, node_ids):
    prompt = f"Extract nodes and relationships from the following text:\n{raw_data}"

    parsed_response = openai_llm_parser(prompt)  # Assuming this returns a list of dictionaries
    parsed_response = parsed_response.graph  # Assuming the 'graph' structure is a key in the parsed response

    nodes = {}
    relationships = []

    for entry in parsed_response:
        node = entry.node
        target_node = entry.target_node  # Get target node if available
        relationship = entry.relationship  # Get relationship if available

        # Reuse the ID of an entity already seen in an earlier paragraph
        nodes[node] = node_ids.setdefault(node, str(uuid.uuid4()))

        if target_node:
            nodes[target_node] = node_ids.setdefault(target_node, str(uuid.uuid4()))

        # Add relationship to the relationships list with node IDs
        if target_node and relationship:
            relationships.append({
                "source": nodes[node],
                "target": nodes[target_node],
                "type": relationship
            })

    return nodes, relationships

def ingest_to_neo4j(nodes, relationships, chunk_id):
    """
    Ingest a paragraph's nodes and relationships into Neo4j,
    linked to a Chunk node that represents the paragraph.
    """

    with neo4j_driver.session() as session:
        session.run("CREATE (c:Chunk {id: $id})", id=chunk_id)

        # Create nodes in Neo4j, once per entity, and link them to the chunk
        for name, node_id in nodes.items():
            session.run(
                "MERGE (n:Entity {id: $id, name: $name}) "
                "WITH n MATCH (c:Chunk {id: $chunk_id}) "
                "CREATE (c)-[:MENTIONS]->(n)",
                id=node_id,
                name=name,
                chunk_id=chunk_id
            )

        # Create relationships in Neo4j
        for relationship in relationships:
            session.run(
                "MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id}) "
                "CREATE (a)-[:RELATIONSHIP {type: $type}]->(b)",
                source_id=relationship["source"],
                target_id=relationship["target"],
                type=relationship["type"]
            )

    return nodes

def create_collection(client, collection_name, vector_dimension):
    if client.collection_exists(collection_name):
        print(f"Skipping creating collection; '{collection_name}' already exists.")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_dimension, distance=models.Distance.COSINE)
    )
    print(f"Collection '{collection_name}' created successfully.")

def openai_embeddings(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    
    return response.data[0].embedding

def ingest_to_qdrant(collection_name, chunk_id, paragraph):
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=chunk_id,
                vector=openai_embeddings(paragraph),
                payload={"id": chunk_id, "text": paragraph}
            )
        ]
    )

def retriever_search(neo4j_driver, qdrant_client, collection_name, query):
    retriever = QdrantNeo4jRetriever(
        driver=neo4j_driver,
        client=qdrant_client,
        collection_name=collection_name,
        id_property_external="id",
        id_property_neo4j="id",
    )

    results = retriever.get_search_results(query_vector=openai_embeddings(query), top_k=5)
    
    return results

def fetch_related_graph(neo4j_client, chunk_ids):
    query = """
    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)-[r1:RELATIONSHIP]-(n1:Entity)-[r2:RELATIONSHIP]-(n2:Entity)
    WHERE c.id IN $chunk_ids
    RETURN e, r1 as r, n1 as related, r2, n2
    UNION
    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)-[r:RELATIONSHIP]-(related:Entity)
    WHERE c.id IN $chunk_ids
    RETURN e, r, related, null as r2, null as n2
    """
    with neo4j_client.session() as session:
        result = session.run(query, chunk_ids=chunk_ids)
        subgraph = []
        for record in result:
            subgraph.append({
                "entity": record["e"],
                "relationship": record["r"],
                "related_node": record["related"]
            })
            if record["r2"] and record["n2"]:
                subgraph.append({
                    "entity": record["related"],
                    "relationship": record["r2"],
                    "related_node": record["n2"]
                })
    return subgraph

def format_graph_context(subgraph):
    nodes = set()
    edges = []

    for entry in subgraph:
        entity = entry["entity"]
        related = entry["related_node"]
        relationship = entry["relationship"]

        nodes.add(entity["name"])
        nodes.add(related["name"])

        edges.append(f"{entity['name']} {relationship['type']} {related['name']}")

    return {"nodes": list(nodes), "edges": list(dict.fromkeys(edges))}  # drop repeated edges

def graphRAG_run(graph_context, user_query):
    nodes_str = ", ".join(graph_context["nodes"])
    edges_str = "; ".join(graph_context["edges"])
    prompt = f"""
    You are an intelligent assistant with access to the following knowledge graph:

    Nodes: {nodes_str}

    Edges: {edges_str}

    Using this graph, Answer the following question:

    User Query: "{user_query}"
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "Provide the answer for the following question:"},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error querying LLM: {str(e)}"
    
if __name__ == "__main__":
    print("Script started")
    print("Creating collection...")
    collection_name = "graphrag"
    vector_dimension = 1536
    create_collection(qdrant_client, collection_name, vector_dimension)
    print("Collection created/verified")
    
    print("Extracting graph components...")
    
    raw_data = """Alice is a data scientist at TechCorp's Seattle office.
    Bob and Carol collaborate on the Alpha project.
    Carol transferred to the New York office last year.
    Dave mentors both Alice and Bob.
    TechCorp's headquarters is in Seattle.
    Carol leads the East Coast team.
    Dave started his career in Seattle.
    The Alpha project is managed from New York.
    Alice previously worked with Carol at DataCo.
    Bob joined the team after Dave's recommendation.
    Eve runs the West Coast operations from Seattle.
    Frank works with Carol on client relations.
    The New York office expanded under Carol's leadership.
    Dave's team spans multiple locations.
    Alice visits Seattle monthly for team meetings.
    Bob's expertise is crucial for the Alpha project.
    Carol implemented new processes in New York.
    Eve and Dave collaborated on previous projects.
    Frank reports to the New York office.
    TechCorp's main AI research is in Seattle.
    The Alpha project revolutionized East Coast operations.
    Dave oversees projects in both offices.
    Bob's contributions are mainly remote.
    Carol's team grew significantly after moving to New York.
    Seattle remains the technology hub for TechCorp."""

    node_ids = {}  # entity name -> ID, shared across paragraphs
    for paragraph in raw_data.split("\n"):
        paragraph = paragraph.strip()
        chunk_id = str(uuid.uuid4())

        nodes, relationships = extract_graph_components(paragraph, node_ids)
        ingest_to_neo4j(nodes, relationships, chunk_id)
        ingest_to_qdrant(collection_name, chunk_id, paragraph)
    print("Ingested", len(node_ids), "entities to Neo4j and Qdrant")

    query = "How is Bob connected to New York?"
    print("Starting retriever search...")
    retriever_result = retriever_search(neo4j_driver, qdrant_client, collection_name, query)
    print("Retriever results:", retriever_result)
    
    print("Extracting chunk IDs...")
    chunk_ids = [record["node"]["id"] for record in retriever_result.records]
    print("Chunk IDs:", chunk_ids)
    
    print("Fetching related graph...")
    subgraph = fetch_related_graph(neo4j_driver, chunk_ids)
    print("Subgraph:", subgraph)
    
    print("Formatting graph context...")
    graph_context = format_graph_context(subgraph)
    print("Graph context:", graph_context)
    
    print("Running GraphRAG...")
    answer = graphRAG_run(graph_context, query)
    print("Final Answer:", answer)
