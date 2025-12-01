import os, json
from clean_text import RAW_DIR
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

def process_all_files() -> List[Dict]:
    """
    Process all text files and create chunks with metadata.
    
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
            text = infile.read()
        
        # Generate chunks using LangChain
        chunks = text_splitter.split_text(text)
        
        for idx, chunk in enumerate(chunks):
            chunk_data = {
                "chunk_id": chunk_id,
                "source_file": filename,
                "chunk_index": idx,
                "text": chunk,
                "char_count": len(chunk)
            }
            all_chunks.append(chunk_data)
            chunk_id += 1
        
        print(f"  Processed {filename}: {len(chunks)} chunks created")
    
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

