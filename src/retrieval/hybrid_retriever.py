import os
import chromadb
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

load_dotenv()

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Vector DB
chroma_client = chromadb.PersistentClient(path="data/vector_db")
collection = chroma_client.get_or_create_collection(name="document_chunks")

# Neo4j
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# ------------------------
# VECTOR SEARCH
# ------------------------
def vector_search(query, top_k=3):
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results["documents"][0]


# ------------------------
# GRAPH SEARCH
# ------------------------
def graph_search(entity_name):
    cypher = """
    MATCH (e:Entity)-[r]-(n:Entity)
    WHERE toLower(e.name) CONTAINS toLower($entity)
    RETURN e.name AS entity, type(r) AS relation, n.name AS neighbor
    LIMIT 10
    """

    with driver.session() as session:
        result = session.run(cypher, entity=entity_name)
        return [dict(record) for record in result]


# ------------------------
# HYBRID RETRIEVAL
# ------------------------
def hybrid_retrieve(query, entity_hint):
    vector_results = vector_search(query)
    graph_results = graph_search(entity_hint)

    return {
        "vector_context": vector_results,
        "graph_context": graph_results
    }


# TEST
if __name__ == "__main__":
    output = hybrid_retrieve(
        query="How does GraphRAG reduce hallucination?",
        entity_hint="GraphRAG"
    )

    print(output)
