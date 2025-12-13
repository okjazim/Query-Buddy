import os
from typing import List
import ollama

from sentence_transformers import SentenceTransformer
from vector_store import search_top_k  

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


# RETRIEVAL
def retrieve(query: str, top_k: int = 3) -> List[dict]:
    """
    Converts query to an embedding and retrieves top-k matching chunks
    from the vector database.
    """
    query_embedding = model.encode([query], normalize_embeddings=True)[0]

    results = search_top_k(query_embedding, top_k)

    return results


# BASELINE LLM ANSWER (NO RAG)

def answer_plain_llm(query: str):
    return ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": query}]
    )["message"]["content"]

# FULL RAG ANSWER
def answer_rag(query: str, top_k: int = 3) -> str:
    """
    Retrieves relevant chunks and injects them into the prompt.
    """

    chunks = retrieve(query, top_k)

    context = "\n\n".join(
        [f"Source {i+1}:\n{c['text']}" for i, c in enumerate(chunks)]
    )

    prompt = f"""
You are an AI assistant answering questions using ONLY the context below.

CONTEXT:
{context}

QUESTION:
{query}

Answer using only the provided context.
"""
    return ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )["message"]["content"]
