import os
import json

import numpy as np
import chromadb
from chromadb.config import Settings

CHROMA_DIR = "./chroma_db"
EMBEDDINGS_PATH = "data/embeddings/embeddings.npy"
METADATA_PATH = "data/embeddings/metadata.json"


# -----------------------------
# Chroma metadata sanitization
# -----------------------------
def _sanitize_metadata_value(v):
    """
    Chroma only allows metadata values that are:
    str, int, float, bool, or None.
    Anything else (dict/list/etc) -> JSON string (or str fallback).
    """
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def _sanitize_metadata(md):
    """
    Ensure each metadata entry is a dict with only primitive values.
    """
    if md is None:
        return {}
    if not isinstance(md, dict):
        return {"_meta": _sanitize_metadata_value(md)}
    return {k: _sanitize_metadata_value(v) for k, v in md.items()}


def load_data():
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"Embeddings file not found at {EMBEDDINGS_PATH}\n Please run embed_store.py first"
        )
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata file not found at {METADATA_PATH}\n Please run embed_store.py first"
        )

    embeddings = np.load(EMBEDDINGS_PATH)
    embeddings_list = embeddings.tolist()

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    texts = [entry["text"] for entry in metadata]
    ids = [f"chunk_{i}" for i in range(len(texts))]

    return texts, embeddings_list, ids, metadata


def store_in_chroma(text_chunks, embedding_vectors, ids, metadata):
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(name="documents_index")

    batch_size = 5000  # safe size under 5461

    for i in range(0, len(text_chunks), batch_size):
        # ✅ sanitize metadata for this batch so Chroma accepts it
        batch_meta_raw = metadata[i:i + batch_size] if metadata else [{} for _ in text_chunks[i:i + batch_size]]
        batch_meta = [_sanitize_metadata(m) for m in batch_meta_raw]

        collection.add(
            documents=text_chunks[i:i + batch_size],
            embeddings=embedding_vectors[i:i + batch_size],
            metadatas=batch_meta,
            ids=ids[i:i + batch_size]
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
