import streamlit as st
from rag_core import answer_plain_llm, answer_rag
import threading

# --- Streamlit App ---

st.title("Query Buddy 🤖")
st.caption("Compare LLM answers with and without RAG.")

with st.sidebar:
    st.header("Search Settings")
    top_k_slider = st.slider(
        "Number of documents for RAG (top_k)",
        min_value=1, max_value=20, value=5, step=1
    )

# Initialize session state for storing results
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
    st.session_state.llm_result = ""
    st.session_state.rag_result = ""

prompt = st.chat_input("Ask a question to compare results...")

if prompt:
    st.session_state.last_query = prompt
    with st.spinner("Generating answers..."):
        # Use a dictionary to store results from threads
        results = {"llm": "", "rag": ""}
 
        # Define target functions for our threads
        def get_llm_answer():
            results["llm"] = answer_plain_llm(prompt)
 
        def get_rag_answer():
            results["rag"] = answer_rag(prompt, top_k=top_k_slider)
 
        # Create and start threads
        llm_thread = threading.Thread(target=get_llm_answer)
        rag_thread = threading.Thread(target=get_rag_answer)
        llm_thread.start()
        rag_thread.start()
 
        # Wait for both threads to complete
        llm_thread.join()
        rag_thread.join()
 
        st.session_state.llm_result = results["llm"]
        st.session_state.rag_result = results["rag"]

# Display results
if st.session_state.last_query:
    st.chat_message("user").markdown(st.session_state.last_query)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("LLM Results (No RAG)")
        if st.session_state.llm_result:
            st.markdown(st.session_state.llm_result)
        else:
            st.info("No response generated.")

    with col2:
        st.subheader("RAG Results")
        if st.session_state.rag_result:
            st.markdown(st.session_state.rag_result)
        else:
            st.info("No response generated.")
else:
    # Initial message on app load
    st.info("Ask a question in the input box below to compare a standard LLM answer with a RAG-powered answer.")
