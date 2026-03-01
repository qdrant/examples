import os
import uuid

from custom_qdrant_neo4j_retriever import CustomQdrantNeo4jRetriever
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient, models

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
        model="gpt-4o-2024-08-06",
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
    
def extract_graph_components(chunks):
    nodes_list = []
    relationships_list = []
    for chunk in chunks:
        prompt = f"Extract nodes and relationships from the following text:\n{chunk}"

        parsed_response = openai_llm_parser(prompt)  # Assuming this returns a list of dictionaries
        parsed_response = parsed_response.graph  # Assuming the 'graph' structure is a key in the parsed response

        nodes = {}
        relationships = []

        for entry in parsed_response:
            node = entry.node
            target_node = entry.target_node  # Get target node if available
            relationship = entry.relationship  # Get relationship if available

            # Add nodes to the dictionary with a unique ID
            if node not in nodes:
                nodes[node] = str(uuid.uuid4())

            if target_node and target_node not in nodes:
                nodes[target_node] = str(uuid.uuid4())

            # Add relationship to the relationships list with node IDs
            if target_node and relationship:
                relationships.append({
                    "source": nodes[node],
                    "target": nodes[target_node],
                    "type": relationship
                })
        
        nodes_list.append(nodes)
        relationships_list.append(relationships)

    return nodes_list, relationships_list

def ingest_to_neo4j(nodes_list, relationships_list):
    """
    Ingest nodes and relationships into Neo4j.
    """

    with neo4j_driver.session() as session:
        # Create nodes in Neo4j
        for nodes in nodes_list:
            for name, node_id in nodes.items():
                session.run(
                    "CREATE (n:Entity {id: $id, name: $name})",
                    id=node_id,
                    name=name
                )

        # Create relationships in Neo4j
        for relationships in relationships_list:
            for relationship in relationships:
                session.run(
                    "MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id}) "
                    "CREATE (a)-[:RELATIONSHIP {type: $type}]->(b)",
                    source_id=relationship["source"],
                    target_id=relationship["target"],
                    type=relationship["type"]
            )

    return nodes_list

def create_collection(client, collection_name, vector_dimension):
  # Try to fetch the collection status
    try:
        collection_info = client.get_collection(collection_name)
        print(f"Skipping creating collection; '{collection_name}' already exists.")
    except Exception as e:
        # If collection does not exist, an error will be thrown, so we create the collection
        if 'Not found: Collection' in str(e):
            print(f"Collection '{collection_name}' not found. Creating it now...")

            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_dimension, distance=models.Distance.COSINE)
            )

            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Error while checking collection: {e}")

def openai_embeddings(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    
    return response.data[0].embedding

def ingest_to_qdrant(collection_name, chunks, node_id_mapping_list):
    embeddings = [openai_embeddings(chunk) for chunk in chunks]

    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            {
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": {"id": list(node_id_mapping.values())}
            }
            for node_id_mapping, embedding in zip(node_id_mapping_list, embeddings)
        ]
    )

def retriever_search(neo4j_driver, qdrant_client, collection_name, query):
    retriever = CustomQdrantNeo4jRetriever(
        driver=neo4j_driver,
        client=qdrant_client,
        collection_name=collection_name,
        id_property_external="id",
        id_property_neo4j="id",
    )

    results = retriever.search(query_vector=openai_embeddings(query), top_k=5)
    
    return results

def fetch_related_graph(neo4j_client, entity_ids):
    query = """
    MATCH (e:Entity)-[r1]-(n1)-[r2]-(n2)
    WHERE e.id IN $entity_ids
    RETURN e, r1 as r, n1 as related, r2, n2
    UNION
    MATCH (e:Entity)-[r]-(related)
    WHERE e.id IN $entity_ids
    RETURN e, r, related, null as r2, null as n2
    """
    with neo4j_client.session() as session:
        result = session.run(query, entity_ids=entity_ids)
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

    return {"nodes": list(nodes), "edges": edges}

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
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Provide the answer for the following question:"},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message
    
    except Exception as e:
        return f"Error querying LLM: {str(e)}"
    
if __name__ == "__main__":
    print("Script started")
    print("Loading environment variables...")
    load_dotenv('.env.local')
    print("Environment variables loaded")
    
    print("Initializing clients...")
    neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
    qdrant_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_key
    )
    print("Clients initialized")
    
    print("Creating collection...")
    collection_name = "graphRAGstoreds"
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

    chunks = raw_data.split("\n")

    nodes_list, relationships_list = extract_graph_components(chunks)
    print("Nodes:", nodes_list)
    print("Relationships:", relationships_list)
    
    print("Ingesting to Neo4j...")
    node_id_mapping_list = ingest_to_neo4j(nodes_list, relationships_list)
    print("Neo4j ingestion complete")
    
    print("Ingesting to Qdrant...")
    ingest_to_qdrant(collection_name, chunks, node_id_mapping_list)
    print("Qdrant ingestion complete")

    query = "How is Bob connected to New York?"
    print("Starting retriever search...")
    retriever_result = retriever_search(neo4j_driver, qdrant_client, collection_name, query)
    print("Retriever results:", retriever_result)
    
    print("Extracting entity IDs...")
    entity_ids = [item.content.split("'id': '")[1].split("'")[0] for item in retriever_result.items]
    print("Entity IDs:", entity_ids)
    
    print("Fetching related graph...")
    subgraph = fetch_related_graph(neo4j_driver, entity_ids)
    print("Subgraph:", subgraph)
    
    print("Formatting graph context...")
    graph_context = format_graph_context(subgraph)
    print("Graph context:", graph_context)
    
    print("Running GraphRAG...")
    answer = graphRAG_run(graph_context, query)
    print("Final Answer:", answer)
