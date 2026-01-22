import os
import json
import numpy as np
import chromadb

EMBEDDINGS_PATH = "data/embeddings/embeddings.npy"
METADATA_PATH = "data/embeddings/metadata.json"
CHROMA_DIR = "./chroma_db"


def load_data():
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError("Embeddings file not found.")
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Metadata file not found.")

    embeddings = np.load(EMBEDDINGS_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    texts = [entry["text"] for entry in metadata]
    ids = [f"doc_{i}" for i in range(len(texts))]
    sources = [entry.get("source_file", "unknown") for entry in metadata]
    return texts, embeddings.tolist(), ids, sources


def store_in_chroma(text_chunks, embedding_vectors, ids):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name="documents_index")

    for i in range(0, len(text_chunks), 5000):
        batch_docs = text_chunks[i:i + 5000]
        batch_embeds = embedding_vectors[i:i + 5000]
        batch_ids = ids[i:i + 5000]
        collection.add(documents=batch_docs, embeddings=batch_embeds, ids=batch_ids)

    print(f"Stored {len(text_chunks)} documents into ChromaDB.")


def load_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name="documents_index")


def search_query(query_text, collection, top_k=5):
    results = collection.query(query_texts=[query_text], n_results=top_k)
    return results


if __name__ == "__main__":
    texts, embeddings, ids, _ = load_data()
    store_in_chroma(texts, embeddings, ids)
