import os
import json

import numpy as np
import chromadb
from chromadb.config import Settings

CHROMA_DIR = "./chroma_db"
EMBEDDINGS_PATH = "data/embeddings/embeddings.npy"
METADATA_PATH = "data/embeddings/metadata.json"


def load_data():
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(f"Embeddings file not found at {EMBEDDINGS_PATH}\n Please run embed_store.py first")
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}\n Please run embed_store.py first")

    embeddings = np.load(EMBEDDINGS_PATH)
    embeddings_list = embeddings.tolist()

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    texts = [entry["text"] for entry in metadata]
    ids = [f"chunk_{i}" for i in range(len(texts))]
    
    # Flatten nested metadata - ChromaDB only accepts flat dicts
    flattened_metadata = []
    for entry in metadata:
        flat_meta = {
            "text": entry.get("text", ""),
            "source_file": entry.get("source_file", ""),
            "modality": entry.get("modality", ""),
            "chunk_index": entry.get("chunk_index", 0),
        }
        
        # Flatten nested "metadata" field if it exists
        if "metadata" in entry and isinstance(entry["metadata"], dict):
            for key, value in entry["metadata"].items():
                # Only add simple types that ChromaDB accepts
                if isinstance(value, (str, int, float, bool)) or value is None:
                    flat_meta[f"meta_{key}"] = value
        
        flattened_metadata.append(flat_meta)

    return texts, embeddings_list, ids, flattened_metadata


def store_in_chroma(text_chunks, embedding_vectors, ids, metadata):

    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name="documents_index")

    batch_size = 5000  # safe size under 5461
    for i in range(0, len(text_chunks), batch_size):
        collection.add(
            documents=text_chunks[i:i+batch_size],
            embeddings=embedding_vectors[i:i+batch_size],
            metadatas=metadata[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )

    print(f"Stored {len(text_chunks)} documents into ChromaDB.")


def search_top_k(query_embedding, top_k=3):
    """
    Searches the Chroma vector database for the top-k most similar chunks.
    Returns matching metadata entries.
    """

    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    collection = client.get_or_create_collection(name="documents_index")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Chroma returns nested lists → flatten them
    matched_metadata = results["metadatas"][0]
    return matched_metadata



if __name__ == "__main__":
    texts, embeddings, ids, metadata = load_data()
    store_in_chroma(texts, embeddings, ids, metadata)
