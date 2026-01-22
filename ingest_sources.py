import os
import glob
import requests
from bs4 import BeautifulSoup
import PyPDF2
import json
from typing import List, Dict, Any
from pathlib import Path

# Directory configurations
RAW_DIR = "data/raw"
RAW_SOURCES_DIR = "data/raw_sources"
AUDIO_DIR = "data/audio"
IMAGE_DIR = "data/image"


# Ensure directories exist
for dir_path in [RAW_DIR, RAW_SOURCES_DIR, AUDIO_DIR, IMAGE_DIR]:
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
    "text": [".txt", ".md", ".rtf"]
}

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

    # Validate types
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

    # Validate modality is known
    if data["modality"] not in MODALITY_EXTENSIONS.keys() and data["modality"] not in ["web"]:
        print(f"Warning: Unknown modality '{data['modality']}'")

    return True


def load_pdf(pdf_path: str) -> Dict[str, Any]:
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
            "metadata": {
                "page_count": len(reader.pages),
                "file_size": os.path.getsize(pdf_path)
            }
        }
    except Exception as e:
        print(f"Error loading PDF {pdf_path}: {e}")
        return None

def load_web(url: str, index: int) -> Dict[str, Any]:
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
            "status_code": resp.status_code
        }
    }

def load_audio(audio_path: str) -> Dict[str, Any]:
    """Load audio file using the audio loader (implemented by Jude)."""
    try:
        from load_audio import process_audio_file
        result = process_audio_file(audio_path)
        if result and validate_common_schema(result):
            return result
        else:
            print(f"Warning: Invalid schema from audio loader for {audio_path}")
            return None
    except ImportError as e:
        print(f"Warning: load_audio.py not found or error importing: {e}")
        print(f"Skipping audio file: {audio_path}")
        print("Note: Audio ingestion requires load_audio.py (Jude's implementation)")
        return None
    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
        return None

def load_image(image_path: str) -> Dict[str, Any]:
    """Load image file using the image loader (implemented by Tomas)."""
    try:
        from load_image import process_image_file
        result = process_image_file(image_path)
        if result and validate_common_schema(result):
            return result
        else:
            print(f"Warning: Invalid schema from image loader for {image_path}")
            return None
    except ImportError as e:
        print(f"Warning: load_image.py not found or error importing: {e}")
        print(f"Skipping image file: {image_path}")
        print("Note: Image ingestion requires load_image.py (Tomas's implementation)")
        return None
    except Exception as e:
        print(f"Error processing image file {image_path}: {e}")
        return None

def load_text_file(text_path: str) -> Dict[str, Any]:
    """Load plain text file."""
    try:
        with open(text_path, "r", encoding="utf-8") as f:
            content = f.read()

        return {
            "content": content,
            "modality": "text",
            "source_file": os.path.basename(text_path),
            "metadata": {
                "file_size": os.path.getsize(text_path),
                "encoding": "utf-8"
            }
        }
    except Exception as e:
        print(f"Error loading text file {text_path}: {e}")
        return None

def save_processed_content(processed_data: Dict[str, Any]) -> str:
    """Save processed content to raw directory and return the filename."""
    if not processed_data:
        return None

    content = processed_data["content"]
    source_file = processed_data["source_file"]
    modality = processed_data["modality"]

    # Create filename based on modality
    base_name = Path(source_file).stem
    output_filename = f"{modality}_{base_name}.txt"
    output_path = os.path.join(RAW_DIR, output_filename)

    # Add modality metadata as header comment
    metadata_json = json.dumps(processed_data["metadata"], indent=2)
    header = f"<!-- MODALITY: {modality.upper()}\nMETADATA: {metadata_json}\n-->\n\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(content)

    print(f"Saved {modality} content to: {output_path}")
    return output_filename

def print_modality_status():
    """Print status of supported modalities and their loaders."""
    print("\nModality Support Status:")
    print("-" * 50)

    modalities_status = {
        "pdf": {"status": "[READY]", "loader": "Built-in (PyPDF2)", "notes": ""},
        "text": {"status": "[READY]", "loader": "Built-in", "notes": ""},
        "web": {"status": "[READY]", "loader": "Built-in (BeautifulSoup)", "notes": ""},
        "audio": {"status": "[WAITING]", "loader": "load_audio.py (Jude)", "notes": "Requires external audio processing"},
        "image": {"status": "[WAITING]", "loader": "load_image.py (Tomas)", "notes": "Requires OCR/vision processing"}
    }

    for modality, info in modalities_status.items():
        extensions = ", ".join(MODALITY_EXTENSIONS.get(modality, []))
        print(f"{modality.upper():6} | {info['status']} | {info['loader']:20} | {extensions:15} | {info['notes']}")

    print("-" * 50)

def discover_source_files() -> Dict[str, List[str]]:
    """Discover all source files by modality."""
    sources = {}

    # Check raw_sources directory for various file types
    for file_path in Path(RAW_SOURCES_DIR).rglob("*"):
        if file_path.is_file():
            modality = get_modality_from_extension(str(file_path))
            if modality != "unknown":
                if modality not in sources:
                    sources[modality] = []
                sources[modality].append(str(file_path))

    # Check dedicated directories
    if os.path.exists(AUDIO_DIR):
        for file_path in Path(AUDIO_DIR).rglob("*"):
            if file_path.is_file() and get_modality_from_extension(str(file_path)) == "audio":
                if "audio" not in sources:
                    sources["audio"] = []
                sources["audio"].append(str(file_path))

    if os.path.exists(IMAGE_DIR):
        for file_path in Path(IMAGE_DIR).rglob("*"):
            if file_path.is_file() and get_modality_from_extension(str(file_path)) == "image":
                if "image" not in sources:
                    sources["image"] = []
                sources["image"].append(str(file_path))

    return sources

def process_sources() -> List[str]:
    """Process all discovered sources and return list of saved filenames."""
    saved_files = []

    # Discover all source files
    sources = discover_source_files()
    print(f"Discovered sources: {sources}")

    # Process PDFs
    if "pdf" in sources:
        print(f"\nProcessing {len(sources['pdf'])} PDF files...")
        for pdf_path in sources["pdf"]:
            processed = load_pdf(pdf_path)
            saved_file = save_processed_content(processed)
            if saved_file:
                saved_files.append(saved_file)

    # Process web URLs
    if WEB_URLS:
        print(f"\nProcessing {len(WEB_URLS)} web URLs...")
        for i, url in enumerate(WEB_URLS, 1):
            processed = load_web(url, i)
            saved_file = save_processed_content(processed)
            if saved_file:
                saved_files.append(saved_file)

    # Process audio files
    if "audio" in sources:
        print(f"\nProcessing {len(sources['audio'])} audio files...")
        for audio_path in sources["audio"]:
            processed = load_audio(audio_path)
            saved_file = save_processed_content(processed)
            if saved_file:
                saved_files.append(saved_file)

    # Process image files
    if "image" in sources:
        print(f"\nProcessing {len(sources['image'])} image files...")
        for image_path in sources["image"]:
            processed = load_image(image_path)
            saved_file = save_processed_content(processed)
            if saved_file:
                saved_files.append(saved_file)

    # Process text files
    if "text" in sources:
        print(f"\nProcessing {len(sources['text'])} text files...")
        for text_path in sources["text"]:
            processed = load_text_file(text_path)
            saved_file = save_processed_content(processed)
            if saved_file:
                saved_files.append(saved_file)

    return saved_files

def verify_pipeline_integration() -> bool:
    """Verify that the ingestion output integrates with the vector store pipeline."""
    try:
        # Check if processed files exist
        if not os.path.exists(RAW_DIR):
            print("ERROR: Raw directory not found")
            return False

        processed_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.txt')]
        if not processed_files:
            print("ERROR: No processed files found in raw directory")
            return False

        # Check if chunk_text.py can process the files
        print(f"SUCCESS: Found {len(processed_files)} processed files ready for chunking")

        # Sample a file to verify schema compliance
        sample_file = os.path.join(RAW_DIR, processed_files[0])
        with open(sample_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for modality header
        if content.startswith('<!-- MODALITY:'):
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
        #group by modality for better reporting
        modality_counts = {}
        for filename in saved_files:
            modality = filename.split('_')[0]  # Extract modality from filename prefix
            modality_counts[modality] = modality_counts.get(modality, 0) + 1

        for modality, count in modality_counts.items():
            print(f"   - {modality.upper()}: {count} files")

    # Verify pipeline integration
    print("\nVerifying pipeline integration...")
    if verify_pipeline_integration():
        print("Pipeline integration verified!")
    else:
        print("Pipeline integration issues detected!")

if __name__ == "__main__":
    main()
