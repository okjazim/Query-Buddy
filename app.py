import os
import streamlit as st
import subprocess
import sys

# --- Constants ---
# Assuming query.py is in the same directory as app.py
QUERY_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "query.py")


# --- Functions ---

def run_query(query_text: str, top_k: int):
    """Runs the query.py script as a subprocess and returns its output."""
    if not os.path.exists(QUERY_SCRIPT_PATH):
        st.error(f"Error: `query.py` not found at `{QUERY_SCRIPT_PATH}`.")
        return None

    try:
        # Ensure we use the same python executable that runs streamlit
        python_executable = sys.executable
        command = [
            python_executable,
            QUERY_SCRIPT_PATH,
            query_text,
            "--top_k",
            str(top_k),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        st.error(f"An error occurred while running the query script:\n\n{e.stderr}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- Streamlit App ---

st.title("Query Buddy 🤖")
st.caption("A simple interface to query your documents using `query.py`.")

with st.sidebar:
    st.header("Search Settings")
    top_k_slider = st.slider(
        "Number of results to return (top_k)",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

# This app is not conversational, so we can simplify to show the last query/response.
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_response" not in st.session_state:
    st.session_state.last_response = ""

# React to user input
prompt = st.chat_input("Ask a question about your documents...")

if prompt:
    st.session_state.last_query = prompt
    with st.spinner("Searching..."):
        response = run_query(prompt, top_k=top_k_slider)
        if response:
            st.session_state.last_response = response
        else:
            st.session_state.last_response = (
                "Failed to get a response from the query script."
            )

# Display the last query and response
if st.session_state.last_query:
    with st.chat_message("user"):
        st.markdown(st.session_state.last_query)
    with st.chat_message("assistant"):
        st.text(st.session_state.last_response)
else:
    # Show an initial message
    with st.chat_message("assistant"):
        st.markdown("How can I help you with your documents?")
