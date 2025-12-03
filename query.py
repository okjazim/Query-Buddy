# query.py
from typing import List, Dict, Any
import argparse

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "documents_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    return collection


def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def similarity_search(query_text: str, top_k: int = 5):

    collection = get_collection()
    model = get_model()

    # Get a NumPy array (shape: (1, dim))
    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True,   # <- changed
    )

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    hits: List[Dict[str, Any]] = []
    docs = results["documents"][0]
    dists = results["distances"][0]
    metas = results["metadatas"][0]
    ids = results["ids"][0]

    for doc_id, doc_text, dist, meta in zip(ids, docs, dists, metas):
        meta = meta or {}  
        hit = {
            "id": doc_id,
            "distance": dist,
            "text": doc_text,
            "metadata": meta,
        }
        hits.append(hit)


    return hits

def cli():
    parser = argparse.ArgumentParser(description="Query the vector store.")
    parser.add_argument("query", type=str, help="The query text to search for.")
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top results to return.",
    )
    args = parser.parse_args()

    hits = similarity_search(args.query, args.top_k)

    print(f"Query: '{args.query}'")
    print(f"Top {args.top_k} results")
    print("-" * 40)

    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        source = meta.get("source_file") or meta.get("source") or "N/A"

        print(f"Result {i}:")
        print(f"  ID      : {hit['id']}")
        print(f"  Distance: {hit['distance']:.4f}")
        print(f"  Source  : {source}")
        print(f"  Text    : {hit['text'][:400]}...")
        print("-" * 40)


if __name__ == "__main__":
    cli()
