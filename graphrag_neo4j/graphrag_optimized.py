from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
from collections import defaultdict
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
import uuid
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import igraph
import leidenalg
import itertools

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

def generate_name_for_text(text, prompt="Give a short descriptive name for this text:"):
    """
    LLM-based name for a single chunk of text. 
    """
    if not text.strip():
        return "NoContent"
    
    completion = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that provides concise names for text content."
            },
            {
                "role": "user",
                "content": f"""{prompt}
                Text:
                {text}

                Short name:"""
            }
        ]
    )
    
    return completion.choices[0].message.content.strip()

def generate_name_for_group(texts, group_prompt="Name this community based on the following texts:"):
    """
    LLM-based name for a group of texts (e.g., a community). 
    """
    if not texts:
        return "EmptyGroup"
    
    snippet = "\n\n".join(texts[:10])  # limit to first 10 to avoid huge prompt
    
    completion = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that provides concise names for groups of related texts."
            },
            {
                "role": "user",
                "content": f"""{group_prompt}
                Here are some representative texts of this group:
                {snippet}

                Short descriptive name for this group:"""
            }
        ]
    )
    
    return completion.choices[0].message.content.strip()
    
def extract_graph_components(raw_data):
    prompt = f"Extract nodes and relationships from the following text:\n{raw_data}"

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

    return nodes, relationships

def ingest_to_neo4j(nodes, relationships):
    """
    Ingest nodes and relationships into Neo4j.
    """

    with neo4j_driver.session() as session:
        # Create nodes in Neo4j
        for name, node_id in nodes.items():
            session.run(
                "CREATE (n:Entity {id: $id, name: $name})",
                id=node_id,
                name=name
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

def ingest_to_qdrant(collection_name, raw_data, node_id_mapping):
    # Split the text into paragraphs and generate embeddings
    paragraphs = raw_data.split("\n")
    embeddings = [openai_embeddings(paragraph) for paragraph in paragraphs]
    
    # Prepare points for Qdrant
    points = []
    for node_id, embedding, paragraph in zip(node_id_mapping.values(), embeddings, paragraphs):
        point_id = str(uuid.uuid4())
        points.append({
            "id": point_id,
            "vector": embedding,
            "payload": {
                "id": node_id,
                "text": paragraph,
                "vector_id": point_id
            }
        })
    
    # Insert points into Qdrant
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    # Return mapping of node_ids to their qdrant point_ids and text
    node_to_qdrant = {
        node_id: {
            "vector_id": point["id"],
            "text": point["payload"]["text"]
        }
        for node_id, point in zip(node_id_mapping.values(), points)
    }
    
    return node_to_qdrant

def detect_communities(node_to_qdrant, collection_name):
    """
    Perform hierarchical community detection on the text embeddings.
    Returns the community labels, super-community labels, and named communities.
    """
    # Get all vectors from Qdrant
    vector_ids = [info["vector_id"] for info in node_to_qdrant.values()]
    
    # Fetch vectors from Qdrant
    vectors = []
    node_ids = []
    node_text_map = {}
    
    # We'll process in batches of 100 to avoid potential issues with large requests
    batch_size = 100
    for i in range(0, len(vector_ids), batch_size):
        batch_ids = vector_ids[i:i+batch_size]
        batch_points = qdrant_client.retrieve(
            collection_name=collection_name,
            ids=batch_ids,
            with_vectors=True,
            with_payload=True
        )
        
        for point in batch_points:
            vectors.append(point.vector)
            node_id = point.payload["id"]
            node_ids.append(node_id)
            node_text_map[node_id] = point.payload["text"]
    
    X = np.array(vectors)
    num_nodes = len(X)
    
    if num_nodes == 0:
        print("No valid nodes found. Exiting community detection.")
        return {}, {}, {}, {}, {}
    
    # ==================== First-Level Community Detection ====================
    k = min(4, num_nodes - 1)  # Ensure k is valid (at least 1 less than num_nodes)
    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine")
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    edges = []
    weights = []
    
    for i in range(num_nodes):
        for j_idx, dist in zip(indices[i], distances[i]):
            if i == j_idx:
                continue
            sim = max(0, 1 - dist)
            edges.append((i, j_idx))
            weights.append(sim)
    
    g = igraph.Graph(n=num_nodes, edges=edges, directed=False)
    g.es["weight"] = weights
    
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=g.es["weight"],
        resolution_parameter=1.0
    )
    community_labels = partition.membership  # each node's first-level community
    
    # Write SIMILAR edges + community label to Neo4j
    with neo4j_driver.session() as session:
        # Clear old SIMILAR edges if any
        session.run("MATCH ()-[r:SIMILAR]->() DELETE r")
        
        # Add similarity relationships
        for i in range(num_nodes):
            this_id = node_ids[i]
            for j_idx, dist in zip(indices[i], distances[i]):
                if i == j_idx:
                    continue
                sim = 1 - dist
                that_id = node_ids[j_idx]
                session.run(
                    """
                    MATCH (a:Entity {id: $idA})
                    MATCH (b:Entity {id: $idB})
                    MERGE (a)-[r:SIMILAR]->(b)
                    SET r.score = $sim
                    """,
                    {"idA": this_id, "idB": that_id, "sim": sim}
                )
        
        # Add community labels
        for i in range(num_nodes):
            this_id = node_ids[i]
            comm = int(community_labels[i])
            session.run(
                """
                MATCH (n:Entity {id: $id})
                SET n.community = $community
                """,
                {"id": this_id, "community": comm}
            )
    
    # ==================== Name first-level communities ====================
    # Group nodes by community
    community_to_nodes = defaultdict(list)
    for i, node_id in enumerate(node_ids):
        comm = community_labels[i]
        community_to_nodes[comm].append(node_id)
    
    # Name each node
    node_names = {}
    for node_id, text in node_text_map.items():
        node_name = generate_name_for_text(text, 
            prompt="Give a short descriptive name for this content:")
        node_names[node_id] = node_name
        
        with neo4j_driver.session() as session:
            session.run(
                """
                MATCH (n:Entity {id: $id})
                SET n.name_generated = $name
                """,
                {"id": node_id, "name": node_name}
            )
    
    # Name each community
    community_name_map = {}
    for comm_id, nodelist in community_to_nodes.items():
        chunk_texts = [node_text_map[nid] for nid in nodelist]
        comm_label = generate_name_for_group(
            chunk_texts,
            group_prompt="Name this first-level community based on its contents:"
        )
        community_name_map[comm_id] = comm_label
        
        with neo4j_driver.session() as session:
            for node_id in nodelist:
                session.run(
                    """
                    MATCH (n:Entity {id: $id})
                    SET n.community_name = $cName
                    """,
                    {"id": node_id, "cName": comm_label}
                )
    
    # ==================== Second-Level Communities ====================
    # Compute centroids for each first-level community
    community_centroids = {}
    for comm_id, nodelist in community_to_nodes.items():
        idxs = [node_ids.index(nid) for nid in nodelist]
        subset_emb = [X[idx] for idx in idxs]
        mean_vec = np.mean(subset_emb, axis=0)
        community_centroids[comm_id] = mean_vec
    
    # Skip super-community detection if we have very few communities
    if len(community_centroids) <= 1:
        return node_text_map, community_labels, {0: 0}, community_name_map, {"0": "All Content"}
    
    comm_ids = sorted(community_centroids.keys())
    comm_index = {c: idx for idx, c in enumerate(comm_ids)}
    num_comms = len(comm_ids)
    
    edges2 = []
    weights2 = []
    
    for c1, c2 in itertools.combinations(comm_ids, 2):
        v1 = community_centroids[c1]
        v2 = community_centroids[c2]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom < 1e-12:
            sim = 0
        else:
            sim = np.dot(v1, v2) / denom
        sim = max(0, sim)
        edges2.append((comm_index[c1], comm_index[c2]))
        weights2.append(sim)
    
    # Skip if we don't have any valid edges between communities
    if not edges2:
        # Single super-community containing all communities
        community_to_super = {c: 0 for c in comm_ids}
        super_community_name_map = {0: "All Content"}
    else:
        g2 = igraph.Graph(n=num_comms, edges=edges2, directed=False)
        g2.es["weight"] = weights2
        
        partition2 = leidenalg.find_partition(
            g2,
            leidenalg.RBConfigurationVertexPartition,
            weights=g2.es["weight"],
            resolution_parameter=1.0
        )
        super_community_labels = partition2.membership
        
        community_to_super = {
            comm_ids[i]: super_community_labels[i] for i in range(num_comms)
        }
        
        # Name the super-communities
        super_comm_to_first_comm = defaultdict(list)
        for fc in comm_ids:
            sc = community_to_super[fc]
            super_comm_to_first_comm[sc].append(fc)
        
        super_community_name_map = {}
        for sc, fc_list in super_comm_to_first_comm.items():
            fc_names = [community_name_map[fc] for fc in fc_list]
            sc_name = generate_name_for_group(
                fc_names,
                group_prompt="Name this second-level community based on the first-level community names:"
            )
            super_community_name_map[sc] = sc_name
    
    # Write super-community info to Neo4j
    with neo4j_driver.session() as session:
        for i in range(num_nodes):
            node_id = node_ids[i]
            fc = community_labels[i]
            sc = community_to_super[fc]
            sc_name = super_community_name_map[sc]
            
            session.run(
                """
                MATCH (n:Entity {id: $id})
                SET n.super_community = $sc,
                    n.super_community_name = $scName
                """,
                {"id": node_id, "sc": sc, "scName": sc_name}
            )
    
    return node_text_map, community_labels, community_to_super, community_name_map, super_community_name_map

def retriever_search(neo4j_driver, qdrant_client, collection_name, query, community_filter=None, super_community_filter=None):
    """
    Enhanced retriever that can filter by community or super-community
    """
    # Get the query embedding
    query_vector = openai_embeddings(query)
    
    # Create the retriever
    retriever = QdrantNeo4jRetriever(
        driver=neo4j_driver,
        client=qdrant_client,
        collection_name=collection_name,
        id_property_external="id",
        id_property_neo4j="id",
    )

    # Get initial results
    results = retriever.search(query_vector=query_vector, top_k=10)
    
    # If filtering by community is requested, filter the results
    if community_filter is not None or super_community_filter is not None:
        # Extract entity IDs from results
        entity_ids = [item.content.split("'id': '")[1].split("'")[0] for item in results.items]
        
        # Query Neo4j to get the community/super-community for each entity
        with neo4j_driver.session() as session:
            records = session.run(
                """
                MATCH (e:Entity)
                WHERE e.id IN $ids
                RETURN e.id as id, e.community as community, e.super_community as super_community
                """,
                {"ids": entity_ids}
            )
            
            # Filter based on community or super-community
            filtered_ids = []
            for record in records:
                if (community_filter is not None and record["community"] == community_filter) or \
                   (super_community_filter is not None and record["super_community"] == super_community_filter):
                    filtered_ids.append(record["id"])
            
            # Re-filter the results
            filtered_results = [item for item in results.items if any(id in item.content for id in filtered_ids)]
            results.items = filtered_results
    
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
    
def process_graph_rag_with_communities(raw_data, query):
    """
    Process text data into a hierarchical knowledge graph with communities,
    then perform GraphRAG with community-aware retrieval.
    """
    print("Script started")
    
    collection_name = "graphRAG_hierarchical"
    vector_dimension = 1536  # OpenAI embedding dimension
    
    print("Creating collection...")
    create_collection(qdrant_client, collection_name, vector_dimension)
    print("Collection created/verified")
    
    print("Extracting graph components...")
    nodes, relationships = extract_graph_components(raw_data)
    print(f"Extracted {len(nodes)} nodes and {len(relationships)} relationships")
    
    print("Ingesting to Neo4j...")
    node_id_mapping = ingest_to_neo4j(nodes, relationships)
    print("Neo4j ingestion complete")
    
    print("Ingesting to Qdrant...")
    node_to_qdrant = ingest_to_qdrant(collection_name, raw_data, node_id_mapping)
    print("Qdrant ingestion complete")
    
    print("Detecting communities...")
    node_text_map, community_labels, community_to_super, community_name_map, super_community_name_map = detect_communities(node_to_qdrant, collection_name)
    print("Community detection complete")
    
    print("Communities:")
    for comm_id, name in community_name_map.items():
        print(f"  - Community {comm_id}: {name}")
    
    print("Super-Communities:")
    for sc_id, name in super_community_name_map.items():
        print(f"  - Super-Community {sc_id}: {name}")
    
    # First try without community filtering
    print("Starting standard retriever search...")
    retriever_result = retriever_search(neo4j_driver, qdrant_client, collection_name, query)
    print(f"Retrieved {len(retriever_result.items)} items")
    
    if retriever_result.items:
        print("Extracting entity IDs...")
        entity_ids = [item.content.split("'id': '")[1].split("'")[0] for item in retriever_result.items]
        print(f"Extracted {len(entity_ids)} entity IDs")
        
        print("Fetching related graph...")
        subgraph = fetch_related_graph(neo4j_driver, entity_ids)
        print(f"Fetched subgraph with {len(subgraph)} edges")
        
        print("Formatting graph context...")
        graph_context = format_graph_context(subgraph)
        print(f"Graph context has {len(graph_context['nodes'])} nodes and {len(graph_context['edges'])} edges")
        
        print("Running standard GraphRAG...")
        standard_answer = graphRAG_run(graph_context, query)
        print("Standard GraphRAG complete")
        
        # Now try with community filtering
        # First, determine the most relevant community
        with neo4j_driver.session() as session:
            records = session.run(
                """
                MATCH (e:Entity)
                WHERE e.id IN $ids
                RETURN e.community as community, count(*) as count
                ORDER BY count DESC
                LIMIT 1
                """,
                {"ids": entity_ids}
            )
            most_relevant_community = None
            for record in records:
                most_relevant_community = record["community"]
                break
        
        if most_relevant_community is not None:
            print(f"Most relevant community: {most_relevant_community} ({community_name_map.get(most_relevant_community, 'Unnamed')})")
            
            print("Starting community-filtered retriever search...")
            community_filtered_result = retriever_search(
                neo4j_driver, 
                qdrant_client, 
                collection_name, 
                query, 
                community_filter=most_relevant_community
            )
            print(f"Retrieved {len(community_filtered_result.items)} items from community {most_relevant_community}")
            
            if community_filtered_result.items:
                print("Extracting community-filtered entity IDs...")
                community_entity_ids = [item.content.split("'id': '")[1].split("'")[0] for item in community_filtered_result.items]
                print(f"Extracted {len(community_entity_ids)} community-filtered entity IDs")
                
                print("Fetching community-filtered related graph...")
                community_subgraph = fetch_related_graph(neo4j_driver, community_entity_ids)
                print(f"Fetched community-filtered subgraph with {len(community_subgraph)} edges")
                
                print("Formatting community-filtered graph context...")
                community_graph_context = format_graph_context(community_subgraph)
                print(f"Community-filtered graph context has {len(community_graph_context['nodes'])} nodes and {len(community_graph_context['edges'])} edges")
                
                print("Running community-filtered GraphRAG...")
                community_answer = graphRAG_run(community_graph_context, query)
                print("Community-filtered GraphRAG complete")
                
                # Return both answers for comparison
                return {
                    "standard_answer": standard_answer.content if hasattr(standard_answer, 'content') else standard_answer,
                    "community_answer": community_answer.content if hasattr(community_answer, 'content') else community_answer,
                    "community_info": {
                        "community_id": most_relevant_community,
                        "community_name": community_name_map.get(most_relevant_community, "Unnamed"),
                        "super_community_id": community_to_super.get(most_relevant_community),
                        "super_community_name": super_community_name_map.get(community_to_super.get(most_relevant_community, 0), "Unnamed")
                    }
                }
            
        return {
            "standard_answer": standard_answer.content if hasattr(standard_answer, 'content') else standard_answer,
            "community_answer": None,
            "community_info": None
        }
    
    return {"standard_answer": "No relevant information found", "community_answer": None, "community_info": None}

if __name__ == "__main__":
    print("Script started")
    print("Loading environment variables...")
    load_dotenv('.env')
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

    nodes, relationships = extract_graph_components(raw_data)
    print("Nodes:", nodes)
    print("Relationships:", relationships)
    
    print("Ingesting to Neo4j...")
    node_id_mapping = ingest_to_neo4j(nodes, relationships)
    print("Neo4j ingestion complete")
    
    print("Ingesting to Qdrant...")
    node_to_qdrant = ingest_to_qdrant(collection_name, raw_data, node_id_mapping)
    print("Qdrant ingestion complete")
    
    print("Detecting communities...")
    node_text_map, community_labels, community_to_super, community_name_map, super_community_name_map = detect_communities(node_to_qdrant, collection_name)
    print("Community detection complete")
    
    print("Communities:")
    for comm_id, name in community_name_map.items():
        print(f"  - Community {comm_id}: {name}")
    
    print("Super-Communities:")
    for sc_id, name in super_community_name_map.items():
        print(f"  - Super-Community {sc_id}: {name}")

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
    
    # Also try with community filtering
    print("\nRerunning with community filtering...")
    
    # Determine most relevant community for the query
    with neo4j_driver.session() as session:
        records = session.run(
            """
            MATCH (e:Entity)
            WHERE e.id IN $ids
            RETURN e.community as community, count(*) as count
            ORDER BY count DESC
            LIMIT 1
            """,
            {"ids": entity_ids}
        )
        most_relevant_community = None
        for record in records:
            most_relevant_community = record["community"]
            break
    
    if most_relevant_community is not None:
        print(f"Most relevant community: {most_relevant_community} ({community_name_map.get(most_relevant_community, 'Unnamed')})")
        
        community_filtered_result = retriever_search(
            neo4j_driver, 
            qdrant_client, 
            collection_name, 
            query, 
            community_filter=most_relevant_community
        )
        
        if community_filtered_result.items:
            community_entity_ids = [item.content.split("'id': '")[1].split("'")[0] for item in community_filtered_result.items]
            community_subgraph = fetch_related_graph(neo4j_driver, community_entity_ids)
            community_graph_context = format_graph_context(community_subgraph)
            
            print("Running community-filtered GraphRAG...")
            community_answer = graphRAG_run(community_graph_context, query)
            print("Final Community-Filtered Answer:", community_answer)
