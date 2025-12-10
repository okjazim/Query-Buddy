# Query-Buddy (Work In Progress)

A simple Retrieval-Augmented Generation (RAG) pipeline made for domain-specific question answering. It utilizes a simple text dataset, which is first cleaned and segmented into optimal chunks. These chunks are converted into dense vector embeddings using a modern transformer model and indexed within a vector database (VD) for high-speed retrieval. The core function involves transforming the user query into a vector, querying the VD for the most relevant document chunks, and using those chunks as contextual grounding to stabilize and inform the LLM's output. This methodology ensures responses are accurate and directly traceable to the input compilation.

## Use Commands in the following order:

```
pip install -r requirements.txt
ollama pull llama3
python main.py
# (in vs-code you can run python line by line by calling it first then calling the following indented)
python
    from rag_core import answer_plain_llm, answer_rag
    answer_plain_llm("when was The Project Gutenberg eBook of Pride and Prejudice released?")
    answer_rag("when was The Project Gutenberg eBook of Pride and Prejudice released?", top_k=3)
```
