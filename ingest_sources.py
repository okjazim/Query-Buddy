# ingest_sources.py
import os
import requests
from bs4 import BeautifulSoup
import PyPDF2
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

# -----------------------------
# Directory configurations
# -----------------------------
RAW_DIR = "data/raw"
RAW_SOURCES_DIR = "data/raw_sources"
AUDIO_DIR = "data/audio"
IMAGE_DIR = "data/image"

# Keep modality outputs organized into subfolders
RAW_SUBDIRS = {
    "pdf": os.path.join(RAW_DIR, "pdf"),
    "web": os.path.join(RAW_DIR, "web"),
    "audio": os.path.join(RAW_DIR, "audio"),
    "image": os.path.join(RAW_DIR, "image"),
    "text": os.path.join(RAW_DIR, "text"),
}

# Ensure directories exist
for dir_path in [RAW_DIR, RAW_SOURCES_DIR, AUDIO_DIR, IMAGE_DIR, *RAW_SUBDIRS.values()]:
    os.makedirs(dir_path, exist_ok=True)

# Web URLs to scrape
WEB_URLS = [
    "https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Global_attributes",
    "https://blog.hubspot.com/website/website-development",
]

# Supported file extensions for each modality
MODALITY_EXTENSIONS = {
    "pdf": [".pdf"],
    "audio": [".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"],
    "image": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"],
    "text": [".txt", ".md", ".rtf"],
}


# -----------------------------
# Helpers
# -----------------------------
def get_modality_from_extension(filename: str) -> str:
    """Determine modality based on file extension."""
    ext = Path(filename).suffix.lower()
    for modality, extensions in MODALITY_EXTENSIONS.items():
        if ext in extensions:
            return modality
    return "unknown"


def validate_common_schema(data: Dict[str, Any]) -> bool:
    """Validate that processed data follows the common schema."""
    required_keys = ["content", "modality", "source_file", "metadata"]

    if not isinstance(data, dict):
        print(f"Error: Expected dict, got {type(data)}")
        return False

    for key in required_keys:
        if key not in data:
            print(f"Error: Missing required key '{key}' in processed data")
            return False

    if not isinstance(data["content"], str):
        print(f"Error: 'content' should be string, got {type(data['content'])}")
        return False
    if not isinstance(data["modality"], str):
        print(f"Error: 'modality' should be string, got {type(data['modality'])}")
        return False
    if not isinstance(data["source_file"], str):
        print(f"Error: 'source_file' should be string, got {type(data['source_file'])}")
        return False
    if not isinstance(data["metadata"], dict):
        print(f"Error: 'metadata' should be dict, got {type(data['metadata'])}")
        return False

    if data["modality"] not in MODALITY_EXTENSIONS.keys() and data["modality"] not in ["web"]:
        print(f"Warning: Unknown modality '{data['modality']}'")

    return True


def _output_path_for(modality: str, source_file: str) -> str:
    """
    Where the processed .txt will be written for this source.
    Mirrors your old naming: {modality}_{stem}.txt but inside data/raw/<modality>/.
    """
    base_name = Path(source_file).stem
    out_name = f"{modality}_{base_name}.txt"
    out_dir = RAW_SUBDIRS.get(modality, RAW_DIR)
    return os.path.join(out_dir, out_name)


def _should_skip(modality: str, source_path: str) -> bool:
    """
    Skip BEFORE doing expensive work.
    - If output doesn't exist -> don't skip
    - If output exists AND is newer than input -> skip
    - If mtime check fails -> skip (safe default)
    """
    source_file = os.path.basename(source_path)
    out_path = _output_path_for(modality, source_file)

    if not os.path.exists(out_path):
        return False

    try:
        return os.path.getmtime(out_path) >= os.path.getmtime(source_path)
    except Exception:
        return True


def _print_skip(modality: str, source_path: str) -> None:
    source_file = os.path.basename(source_path)
    out_path = _output_path_for(modality, source_file)
    print(f"⏭️  Skipping (already exists): {out_path}")


# -----------------------------
# Loaders
# -----------------------------
def load_pdf(pdf_path: str) -> Optional[Dict[str, Any]]:
    """Load PDF and return text content with metadata."""
    try:
        text = ""
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                text += f"[Page {page_num + 1}]\n{page_text}\n\n"

        return {
            "content": text,
            "modality": "pdf",
            "source_file": os.path.basename(pdf_path),
            "metadata": {"page_count": len(reader.pages), "file_size": os.path.getsize(pdf_path)},
        }
    except Exception as e:
        print(f"Error loading PDF {pdf_path}: {e}")
        return None


def load_web(url: str, index: int) -> Optional[Dict[str, Any]]:
    """Load web page and return text content with metadata."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Error loading web page {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text(separator="\n")

    return {
        "content": text,
        "modality": "web",
        "source_file": f"web_{index}.txt",
        "metadata": {
            "url": url,
            "title": soup.title.string if soup.title else "No title",
            "status_code": resp.status_code,
        },
    }


def load_audio(audio_path: str) -> Optional[Dict[str, Any]]:
    """Load audio file using the audio loader."""
    try:
        from load_audio import process_audio_file

        result = process_audio_file(audio_path)
        if result and validate_common_schema(result):
            return result

        print(f"Warning: Invalid schema from audio loader for {audio_path}")
        return None

    except ImportError as e:
        print(f"Warning: load_audio.py not found or error importing: {e}")
        print(f"Skipping audio file: {audio_path}")
        return None
    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
        return None


def load_image(image_path: str) -> Optional[Dict[str, Any]]:
    """Load image file using the image loader."""
    try:
        from load_image import process_image_file

        result = process_image_file(image_path)
        if result and validate_common_schema(result):
            return result

        print(f"Warning: Invalid schema from image loader for {image_path}")
        return None

    except ImportError as e:
        print(f"Warning: load_image.py not found or error importing: {e}")
        print(f"Skipping image file: {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image file {image_path}: {e}")
        return None


def load_text_file(text_path: str) -> Optional[Dict[str, Any]]:
    """Load plain text file."""
    try:
        with open(text_path, "r", encoding="utf-8") as f:
            content = f.read()

        return {
            "content": content,
            "modality": "text",
            "source_file": os.path.basename(text_path),
            "metadata": {"file_size": os.path.getsize(text_path), "encoding": "utf-8"},
        }
    except Exception as e:
        print(f"Error loading text file {text_path}: {e}")
        return None


# -----------------------------
# Saving
# -----------------------------
def save_processed_content(processed_data: Dict[str, Any]) -> Optional[str]:
    """Save processed content to raw directory and return the filename."""
    if not processed_data:
        return None

    content = processed_data["content"]
    source_file = processed_data["source_file"]
    modality = processed_data["modality"]

    output_path = _output_path_for(modality, source_file)
    output_filename = os.path.basename(output_path)

    # Add modality metadata as header comment
    metadata_json = json.dumps(processed_data["metadata"], indent=2)
    header = f"<!-- MODALITY: {modality.upper()}\nMETADATA: {metadata_json}\n-->\n\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(content)

    print(f"Saved {modality} content to: {output_path}")
    return output_filename


# -----------------------------
# Status / discovery
# -----------------------------
def print_modality_status():
    """Print status of supported modalities and their loaders."""
    print("\nModality Support Status:")
    print("-" * 70)

    modalities_status = {
        "pdf": {"status": "[READY]", "loader": "Built-in (PyPDF2)", "notes": ""},
        "text": {"status": "[READY]", "loader": "Built-in", "notes": ""},
        "web": {"status": "[READY]", "loader": "Built-in (BeautifulSoup)", "notes": ""},
        "audio": {"status": "[READY]", "loader": "load_audio.py", "notes": "Requires transcription backend"},
        "image": {"status": "[READY]", "loader": "load_image.py", "notes": "Requires OCR/vision processing"},
    }

    for modality, info in modalities_status.items():
        extensions = ", ".join(MODALITY_EXTENSIONS.get(modality, []))
        print(
            f"{modality.upper():6} | {info['status']} | {info['loader']:24} | {extensions:30} | {info['notes']}"
        )

    print("-" * 70)


def discover_source_files() -> Dict[str, List[str]]:
    """Discover all source files by modality."""
    sources: Dict[str, List[str]] = {}

    # raw_sources directory
    for file_path in Path(RAW_SOURCES_DIR).rglob("*"):
        if file_path.is_file():
            modality = get_modality_from_extension(str(file_path))
            if modality != "unknown":
                sources.setdefault(modality, []).append(str(file_path))

    # dedicated dirs
    if os.path.exists(AUDIO_DIR):
        for file_path in Path(AUDIO_DIR).rglob("*"):
            if file_path.is_file() and get_modality_from_extension(str(file_path)) == "audio":
                sources.setdefault("audio", []).append(str(file_path))

    if os.path.exists(IMAGE_DIR):
        for file_path in Path(IMAGE_DIR).rglob("*"):
            if file_path.is_file() and get_modality_from_extension(str(file_path)) == "image":
                sources.setdefault("image", []).append(str(file_path))

    # deterministic order
    for k in list(sources.keys()):
        sources[k].sort()

    return sources


# -----------------------------
# Processing
# -----------------------------
def process_sources() -> List[str]:
    """Process all discovered sources and return list of saved filenames."""
    saved_files: List[str] = []

    sources = discover_source_files()
    print(f"Discovered sources: {sources}")

    # PDFs
    if "pdf" in sources:
        print(f"\nProcessing {len(sources['pdf'])} PDF files...")
        for pdf_path in sources["pdf"]:
            if _should_skip("pdf", pdf_path):
                _print_skip("pdf", pdf_path)
                continue
            processed = load_pdf(pdf_path)
            saved = save_processed_content(processed) if processed else None
            if saved:
                saved_files.append(saved)

    # Web
    if WEB_URLS:
        print(f"\nProcessing {len(WEB_URLS)} web URLs...")
        for i, url in enumerate(WEB_URLS, 1):
            # web "input mtime" doesn't exist; just skip if output exists
            out_path = _output_path_for("web", f"web_{i}.txt")
            if os.path.exists(out_path):
                print(f"⏭️  Skipping (already exists): {out_path}")
                continue
            processed = load_web(url, i)
            saved = save_processed_content(processed) if processed else None
            if saved:
                saved_files.append(saved)

    # Audio (expensive) -> SKIP CHECK FIRST
    if "audio" in sources:
        print(f"\nProcessing {len(sources['audio'])} audio files...")
        for audio_path in sources["audio"]:
            if _should_skip("audio", audio_path):
                _print_skip("audio", audio_path)
                continue
            processed = load_audio(audio_path)  # only transcribes if not skipped
            saved = save_processed_content(processed) if processed else None
            if saved:
                saved_files.append(saved)

    # Images (expensive) -> SKIP CHECK FIRST (plus your load_image cache can still exist)
    if "image" in sources:
        print(f"\nProcessing {len(sources['image'])} image files...")
        for image_path in sources["image"]:
            if _should_skip("image", image_path):
                _print_skip("image", image_path)
                continue
            processed = load_image(image_path)
            saved = save_processed_content(processed) if processed else None
            if saved:
                saved_files.append(saved)

    # Text
    if "text" in sources:
        print(f"\nProcessing {len(sources['text'])} text files...")
        for text_path in sources["text"]:
            if _should_skip("text", text_path):
                _print_skip("text", text_path)
                continue
            processed = load_text_file(text_path)
            saved = save_processed_content(processed) if processed else None
            if saved:
                saved_files.append(saved)

    return saved_files


def verify_pipeline_integration() -> bool:
    """Verify that the ingestion output integrates with the vector store pipeline."""
    try:
        if not os.path.exists(RAW_DIR):
            print("ERROR: Raw directory not found")
            return False

        # gather .txt from subdirs
        processed_files: List[str] = []
        for sub in RAW_SUBDIRS.values():
            if os.path.exists(sub):
                processed_files.extend([os.path.join(sub, f) for f in os.listdir(sub) if f.endswith(".txt")])

        if not processed_files:
            print("ERROR: No processed files found in raw directory")
            return False

        print(f"SUCCESS: Found {len(processed_files)} processed files ready for chunking")

        sample_file = processed_files[0]
        with open(sample_file, "r", encoding="utf-8") as f:
            content = f.read()

        if content.startswith("<!-- MODALITY:"):
            print("SUCCESS: Files contain proper modality metadata headers")
        else:
            print("WARNING: Files may not have modality headers (older format?)")

        return True

    except Exception as e:
        print(f"ERROR: Pipeline verification failed: {e}")
        return False


def main():
    """Main ingestion orchestration function."""
    print("Starting multimodal ingestion pipeline...")
    print_modality_status()

    saved_files = process_sources()

    print(f"\nIngestion complete!")
    print(f"Processed {len(saved_files)} files saved to {RAW_DIR}")

    if saved_files:
        print("Files processed:")
        modality_counts: Dict[str, int] = {}
        for filename in saved_files:
            modality = filename.split("_")[0]
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
        for modality, count in modality_counts.items():
            print(f"   - {modality.upper()}: {count} files")

    print("\nVerifying pipeline integration...")
    if verify_pipeline_integration():
        print("Pipeline integration verified!")
    else:
        print("Pipeline integration issues detected!")


if __name__ == "__main__":
    main()
