import os
import streamlit as st
from typing import List, Dict, Any

# Import the necessary functions from your query script
from query import get_model, get_collection
from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection


# --- Functions ---

@st.cache_resource
def load_resources():
    """
    Loads the embedding model and ChromaDB collection using functions from query.py.
    Caches the resources to avoid reloading them on every query.
    """
    try:
        model = get_model()
        collection = get_collection()
        return model, collection
    except Exception as e:
        st.error(
            f"Failed to load model or database. Have you run `embed_store.py` and `vector_store.py` first?\n\nError: {e}"
        )
        st.stop()


def perform_search(
    query_text: str,
    model: SentenceTransformer,
    collection: Collection,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Performs a similarity search on the vector store.
    This logic is adapted from query.py's similarity_search function for direct use.
    """
    query_embedding = model.encode([query_text], normalize_embeddings=True)

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )

    hits: List[Dict[str, Any]] = []
    if not results or not results.get("ids") or not results["ids"][0]:
        return []

    # Unpack results
    ids = results["ids"][0]
    docs = results["documents"][0]
    dists = results["distances"][0]
    metas = results["metadatas"][0]

    for doc_id, doc_text, dist, meta in zip(ids, docs, dists, metas):
        hit = {
            "id": doc_id,
            "distance": dist,
            "text": doc_text,
            "metadata": meta or {},
        }
        hits.append(hit)

    return hits


# --- Streamlit App ---

st.title("Query Buddy 🤖")
st.caption("An efficient interface to query your documents.")

# Load resources once and cache them
model, collection = load_resources()

with st.sidebar:
    st.header("Search Settings")
    top_k_slider = st.slider(
        "Number of results to return (top_k)",
        min_value=1, max_value=20, value=5, step=1
    )

# Initialize a chat-like history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you with your documents?"}]

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Handle rendering of search results (list) vs. text (str)
        if isinstance(message["content"], list):
            for i, hit in enumerate(message["content"], start=1):
                meta = hit.get("metadata", {})
                source = meta.get("source_file", "N/A")
                with st.expander(f"**Result {i}:** `{source}` (Distance: {hit['distance']:.4f})"):
                    st.markdown(hit["text"])
        else:
            st.markdown(message["content"])

# React to user input
prompt = st.chat_input("Ask a question about your documents...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            hits = perform_search(prompt, model, collection, top_k=top_k_slider)
            if not hits:
                response = "I couldn't find any relevant documents for your query."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                # Display results and add them to history as a list of dicts
                for i, hit in enumerate(hits, start=1):
                    meta = hit.get("metadata", {})
                    source = meta.get("source_file", "N/A")
                    with st.expander(f"**Result {i}:** `{source}` (Distance: {hit['distance']:.4f})"):
                        st.markdown(hit["text"])
                st.session_state.messages.append({"role": "assistant", "content": hits})
else:
    st.session_state.last_response = (
    "Failed to get a response from the query script.")

    # # Display the last query and response
    # if st.session_state.last_query:
    #     with st.chat_message("user"):
    #         st.markdown(st.session_state.last_query)
    #     with st.chat_message("assistant"):
    #         st.text(st.session_state.last_response)
    # else:
    #     # Show an initial message
    #     with st.chat_message("assistant"):
    #         st.markdown("How can I help you with your documents?")
