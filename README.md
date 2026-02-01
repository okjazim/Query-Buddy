# Query-Buddy

---

## Project Description

Query-Buddy is a Retrieval-Augmented Generation (RAG) pipeline for domain-specific question answering. It ingests multiple data formats (text, PDF, images, audio), cleans and segments content into chunks, converts them into dense vector embeddings, and indexes them in a vector database. User queries are embedded, matched against the index for the most relevant chunks, and those chunks are used as context so the LLM produces accurate, traceable answers grounded in your data.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Ingestion** | PyPDF2, BeautifulSoup (web), Whisper (audio), Tesseract/Pillow (images) |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) |
| **Vector DB** | ChromaDB |
| **LLM** | Ollama (e.g. Llama 3) |
| **UI** | Streamlit |

---

## Features

- **Multimodal ingestion**: PDF, plain text, web pages, images (OCR/captioning), and audio (ASR/transcription)
- **Unified schema**: All modalities produce text + metadata in a common format for chunking and embedding
- **RAG pipeline**: Ingest → Chunk → Embed → Index, orchestrated via `main.py`
- **Query interface**: Web UI (Streamlit) and Python API to compare plain LLM vs RAG answers
- **Modality-aware metadata**: Chunks retain source type and file metadata for filtering and citations

---

## Setup Instructions

1. **Clone the repository** (or use your local copy):
   ```bash
   git clone <repository-url>
   cd Query-Buddy
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama and pull the LLM** (for local LLM answers):
   ```bash
   ollama pull llama3
   ```

5. **Optional – system dependencies for full multimodal support**:
   - **Tesseract OCR**: Required for image text extraction. Install from [Tesseract](https://github.com/tesseract-ocr/tesseract) and ensure it is on your PATH.
   - **FFmpeg**: Often needed for Whisper/audio processing. Install per your OS.

---

## Usage

**Run the full pipeline** (ingest → chunk → embed → index):

```bash
python main.py
```

**Skip ingestion** (if data is already in `data/raw`):

```bash
python main.py --skip-ingest
```

**Run the Streamlit app** (compare RAG vs plain LLM):

```bash
streamlit run app.py
```

**Query from Python**:

```python
from rag_core import answer_plain_llm, answer_rag

# RAG answer (uses your indexed documents)
answer_rag("Your question here?", top_k=3)

# Plain LLM answer (no retrieval)
answer_plain_llm("Your question here?")
```

**Data locations**:

- Put source files in `data/raw_sources/` or in `data/audio/` and `data/image/` for dedicated modalities. The ingestion step discovers and processes them.

---

## Screenshots / Video Demo

_Add screenshots of the Streamlit app or a short video demo here._

Example placeholder:
```
[Screenshot: Streamlit Query Buddy UI]
[Video: Link to demo video if available]
```

---

## File Structure

```
Query-Buddy/
├── main.py              # Pipeline orchestrator (ingest → chunk → embed → index)
├── ingest_sources.py    # Central ingestion dispatcher (PDF, text, web, audio, image)
├── load_audio.py        # Audio transcription (e.g. Whisper)
├── load_image.py        # Image OCR / captioning (e.g. Tesseract)
├── chunk_text.py        # Text chunking with modality metadata
├── embed_store.py       # Embedding generation and metadata export
├── vector_store.py      # ChromaDB indexing and retrieval
├── rag_core.py          # RAG retrieval and LLM answering
├── query.py             # Query interface (e.g. filtering, citations)
├── app.py               # Streamlit web UI
├── requirements.txt    # Python dependencies
├── data/
│   ├── raw/             # Processed text output from ingestion
│   ├── raw_sources/     # Source PDFs, text files, etc.
│   ├── audio/           # Audio files
│   ├── image/           # Image files (if used)
│   ├── chunks.json      # Chunks produced by chunk_text.py
│   └── embeddings/      # Embeddings and metadata for vector_store
├── chroma_db/           # ChromaDB persistence (created at runtime)
└── README.md
```

---

## Contribution

Contributions are welcome. Suggested workflow:

1. Fork the repository.
2. Create a branch for your feature or fix.
3. Make changes and add tests if applicable.
4. Open a pull request with a short description of the change.

For larger features (e.g. new modalities or loaders), open an issue first to align with the project structure and common schema.

---

## Important Note

- **Ollama** must be installed and running locally for LLM answers. Ensure `ollama pull llama3` (or your chosen model) has been run.
- **First run**: Running `python main.py` without existing data will ingest from configured web URLs and any files in `data/raw_sources/`, `data/audio/`, and `data/image/`. Place your documents there before ingestion if you want to query your own content.
- **Multimodal loaders**: Audio and image ingestion depend on `load_audio.py` and `load_image.py` (and their system dependencies). If those modules or dependencies are missing, ingestion skips those modalities and continues with the rest.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for the full text.

---

## References

- [Ollama](https://ollama.ai/) – Local LLM runtime
- [ChromaDB](https://www.trychroma.com/) – Vector database
- [Sentence Transformers](https://www.sbert.net/) – Embedding models
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/) – Document chunking
- [Streamlit](https://streamlit.io/) – Web app framework
