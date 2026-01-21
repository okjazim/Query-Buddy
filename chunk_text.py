import os, json, re
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

CHUNKS_OUTPUT = "data/chunks.json"

# Chunking parameters
CHUNK_SIZE = 500      # Number of characters per chunk
CHUNK_OVERLAP = 50    # Number of overlapping characters between chunks

def initialize_text_splitter(chunk_size: int = CHUNK_SIZE, 
                             chunk_overlap: int = CHUNK_OVERLAP) -> RecursiveCharacterTextSplitter:
    """
    Initialize the LangChain text splitter with specified parameters.
    
    Args:
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
    
    Returns:
        Configured RecursiveCharacterTextSplitter instance
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

def parse_file_header(text: str) -> tuple[str, Dict[str, Any]]:
    """
    Parse modality and metadata from file header comments.

    Returns:
        Tuple of (modality, metadata_dict)
    """
    modality = "text"  # default
    metadata = {}

    # Look for header comment pattern
    header_pattern = r"<!-- MODALITY: (\w+)\s*METADATA: (\{.*?\})\s*-->"
    match = re.search(header_pattern, text, re.DOTALL)

    if match:
        modality = match.group(1).lower()
        try:
            metadata = json.loads(match.group(2))
        except json.JSONDecodeError:
            print(f"Warning: Could not parse metadata JSON: {match.group(2)}")

    return modality, metadata

def process_all_files() -> List[Dict]:
    """
    Process all text files and create chunks with multimodal metadata.

    Returns:
        List of dictionaries containing chunk text and metadata
    """
    all_chunks = []
    chunk_id = 0

    # Initialize the text splitter
    text_splitter = initialize_text_splitter()

    # Get all cleaned text files
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".txt")]

    if not files:
        print(f"Warning: No files found in {RAW_DIR}")
        return all_chunks

    print(f"Found {len(files)} files to process...")

    for filename in sorted(files):
        filepath = os.path.join(RAW_DIR, filename)

        # Read cleaned text
        with open(filepath, "r", encoding="utf-8") as infile:
            full_text = infile.read()

        # Parse modality and metadata from header
        modality, file_metadata = parse_file_header(full_text)

        # Remove header from text for chunking
        text_to_chunk = re.sub(r"<!--.*?-->\s*", "", full_text, flags=re.DOTALL).strip()

        # Generate chunks using LangChain
        chunks = text_splitter.split_text(text_to_chunk)

        for idx, chunk in enumerate(chunks):
            chunk_data = {
                "chunk_id": chunk_id,
                "source_file": filename,
                "chunk_index": idx,
                "text": chunk,
                "char_count": len(chunk),
                "modality": modality,
                "file_metadata": file_metadata
            }
            all_chunks.append(chunk_data)
            chunk_id += 1

        print(f"  Processed {filename} ({modality}): {len(chunks)} chunks created")

    return all_chunks

def save_chunks(chunks: List[Dict], output_path: str = CHUNKS_OUTPUT):
    """
    Save chunks to a JSON file.
    
    Args:
        chunks: List of chunk dictionaries
        output_path: Path to save the JSON file
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(chunks, outfile, indent=2, ensure_ascii=False)
    
    print(f"\nChunks saved to {output_path}")

# Process all files and create chunks
chunks = process_all_files()
    
if chunks:
    # Save chunks to JSON
    save_chunks(chunks)

# Show a sample chunk
    print("\nSample chunk (first chunk):")
    print("-" * 60)
    print(f"Chunk ID: {chunks[0]['chunk_id']}")
    print(f"Source: {chunks[0]['source_file']}")
    print(f"Text preview: {chunks[0]['text'][:200]}...")
    print("-" * 60)
else:
    print("No chunks were created. Please check your input files.")

