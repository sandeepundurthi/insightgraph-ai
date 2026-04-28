import json
import chromadb
from sentence_transformers import SentenceTransformer


def build_vector_store(chunks_file="data/chunks/chunks.json"):
    # Load chunks
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks")

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create persistent Chroma DB
    client = chromadb.PersistentClient(path="data/vector_db")
    collection = client.get_or_create_collection(name="document_chunks")

    for chunk in chunks:
        embedding = model.encode(chunk["text"]).tolist()

        collection.upsert(
            ids=[chunk["chunk_id"]],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{"source": chunk["source"]}]
        )

    print("✅ Vector store created successfully!")


if __name__ == "__main__":
    build_vector_store()
