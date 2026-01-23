# chunk_text.py
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

RAW_DIR = "data/raw"
CHUNKS_OUTPUT = "data/chunks.json"

# Chunking parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

os.makedirs(RAW_DIR, exist_ok=True)


def initialize_text_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def parse_file_header(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Parse modality + metadata from header like:

    <!-- MODALITY: IMAGE
    METADATA: {...}
    -->
    """
    modality = "text"
    metadata: Dict[str, Any] = {}

    header_pattern = r"<!--\s*MODALITY:\s*(\w+)\s*METADATA:\s*(\{.*?\})\s*-->"
    match = re.search(header_pattern, text, re.DOTALL)

    if match:
        modality = (match.group(1) or "text").lower()
        meta_str = match.group(2) or "{}"
        try:
            metadata = json.loads(meta_str)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse metadata JSON in header (using empty dict).")

    return modality, metadata


def _strip_header(text: str) -> str:
    return re.sub(r"<!--.*?-->\s*", "", text, flags=re.DOTALL).strip()


def discover_raw_txt_files(raw_dir: str = RAW_DIR) -> List[str]:
    """
    Recursively find ALL .txt files under data/raw, including subfolders like:
      data/raw/pdf/*.txt
      data/raw/web/*.txt
      data/raw/audio/*.txt
      data/raw/image/*.txt
    """
    root = Path(raw_dir)
    if not root.exists():
        return []
    return sorted([str(p) for p in root.rglob("*.txt")])


def process_all_files(raw_dir: str = RAW_DIR) -> List[Dict]:
    """
    Chunk all processed raw files (recursively) into data/chunks.json
    while preserving modality + file_metadata from the header.
    """
    all_chunks: List[Dict] = []
    chunk_id = 0

    text_splitter = initialize_text_splitter()

    filepaths = discover_raw_txt_files(raw_dir)
    if not filepaths:
        print(f"Warning: No .txt files found under {raw_dir}")
        return all_chunks

    print(f"Found {len(filepaths)} files to process...")

    for filepath in filepaths:
        # Keep subfolder in name for uniqueness + filtering later
        # (Windows example: "image\\image_005_Goblin-shark.txt")
        source_file = os.path.relpath(filepath, raw_dir)

        try:
            with open(filepath, "r", encoding="utf-8") as infile:
                full_text = infile.read()
        except Exception as e:
            print(f"[chunk_text] Skipping {source_file} (read error): {e}")
            continue

        modality, file_metadata = parse_file_header(full_text)
        text_to_chunk = _strip_header(full_text)

        if not text_to_chunk.strip():
            print(f"[chunk_text] Skipping {source_file} (empty content after header strip)")
            continue

        chunks = text_splitter.split_text(text_to_chunk)

        for idx, chunk in enumerate(chunks):
            chunk_data = {
                "chunk_id": chunk_id,
                "source_file": source_file,
                "chunk_index": idx,
                "text": chunk,
                "char_count": len(chunk),
                "modality": modality,
                "file_metadata": file_metadata,
            }
            all_chunks.append(chunk_data)
            chunk_id += 1

        print(f"  Processed {source_file} ({modality}): {len(chunks)} chunks created")

    return all_chunks


def save_chunks(chunks: List[Dict], output_path: str = CHUNKS_OUTPUT) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(chunks, outfile, indent=2, ensure_ascii=False)
    print(f"\nChunks saved to {output_path}")


def main():
    chunks = process_all_files(RAW_DIR)
    if chunks:
        save_chunks(chunks, CHUNKS_OUTPUT)

        print("\nSample chunk (first chunk):")
        print("-" * 60)
        print(f"Chunk ID: {chunks[0]['chunk_id']}")
        print(f"Source: {chunks[0]['source_file']}")
        print(f"Modality: {chunks[0]['modality']}")
        print(f"Text preview: {chunks[0]['text'][:200]}...")
        print("-" * 60)
    else:
        print("No chunks were created. Please check your input files.")


if __name__ == "__main__":
    main()
