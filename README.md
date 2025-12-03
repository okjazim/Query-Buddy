# Query-Buddy
A simple Retrieval-Augmented Generation (RAG) pipeline made for domain-specific question answering. It utilizes a simple text dataset, which is first cleaned and segmented into optimal chunks. These chunks are converted into dense vector embeddings using a modern transformer model and indexed within a vector database (VD) for high-speed retrieval. The core function involves transforming the user query into a vector, querying the VD for the most relevant document chunks, and using those chunks as contextual grounding to stabilize and inform the LLM's output. This methodology ensures responses are accurate and directly traceable to the input compilation.

## Use Commands in the following order:
```
python clean_text.py
python chunk_text.py 
python embed_store.py 
python vector_store.py
python query.py "{Write text to search here}" --top_k 5
```
