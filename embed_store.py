import os
import json
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/chunks.json"          
EMBEDDINGS_DIR = "data/embeddings"      
EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "metadata.json")

#model for generating embeddings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(path: str = CHUNKS_PATH) -> List[Dict]:
    """Load chunk dictionaries produced by chunk text"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run chunk_text.py first to generate chunks."
        )

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Loaded chunks.json is empty or not a list.")

    print(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load the transformers model used for embeddings."""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    return model


def create_embeddings(
    chunks: List[Dict],
    model: SentenceTransformer,
) -> np.ndarray:
    """Generate embeddings for each chunk's text."""
    texts = [c["text"] for c in chunks]

    print(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # good for cosine similarity later
    )

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def save_embeddings_and_metadata(
    embeddings: np.ndarray,
    chunks: List[Dict],
    embeddings_path: str = EMBEDDINGS_PATH,
    metadata_path: str = METADATA_PATH,
):
    """Save embeddings as .npy and metadata as JSON."""
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)

    # make vectors
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path}")

    # metadata 
    metadata = []
    for c in chunks:
        metadata.append(
            {
                "chunk_id": c["chunk_id"],
                "source_file": c["source_file"],
                "chunk_index": c["chunk_index"],
                "char_count": c["char_count"],
                "text": c["text"],  # emergency blakcbox thingy
            }
        )

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved metadata to {metadata_path}")
    print(f"Total chunks: {len(metadata)}")


def main():
    chunks = load_chunks()
    model = load_model()
    embeddings = create_embeddings(chunks, model)
    save_embeddings_and_metadata(embeddings, chunks)


if __name__ == "__main__":
    main()
