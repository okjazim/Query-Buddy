from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# -----------------------------
# Configuration
# -----------------------------

SUPPORTED_AUDIO_EXTS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".webm", ".mp4", ".mkv"
}

DEFAULT_DATA_DIR = "data"

# Controls optional re-chunking of long segments
DEFAULT_MAX_CHARS_PER_CHUNK = 900 # keep chunks reasonably sized for embedding/RAG
DEFAULT_OVERLAP_CHARS = 80 # overlap helps avoid cutting off meaning


@dataclass
class AudioChunk:
    text: str
    source_file: str
    chunk_index: int
    char_count: int

    # extra metadata (optional but useful)
    modality: str = "audio"
    source_path: Optional[str] = None
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    segment_index: Optional[int] = None
    language: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
            "modality": self.modality,
            "source_path": self.source_path,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "segment_index": self.segment_index,
            "language": self.language,
        }


# -----------------------------
# Utility: file discovery
# -----------------------------

def find_audio_files(root_dir: str = DEFAULT_DATA_DIR) -> List[str]:
    """
    Recursively discover audio files under root_dir.

    Returns:
        List of file paths (strings), sorted for deterministic behavior.
    """
    root = Path(root_dir)
    if not root.exists():
        return []

    files: List[str] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS:
            files.append(str(p))
    files.sort()
    return files


# -----------------------------
# Utility: chunking text
# -----------------------------

def _chunk_text_by_chars(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    overlap: int = DEFAULT_OVERLAP_CHARS
) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        # move start forward but keep overlap
        start = max(0, end - overlap)

    return chunks


# -----------------------------
# Transcription backend selection
# -----------------------------

def _load_faster_whisper():
    """Try to load faster-whisper. Returns WhisperModel class or None."""
    try:
        from faster_whisper import WhisperModel # type: ignore
        return WhisperModel
    except Exception:
        return None


def _load_openai_whisper():
    """Try to load openai-whisper. Returns whisper module or None."""
    try:
        import whisper # type: ignore
        return whisper
    except Exception:
        return None


# -----------------------------
# Transcription: faster-whisper (preferred)
# -----------------------------

def _transcribe_with_faster_whisper(
    audio_path: str,
    model_name: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    language: Optional[str] = None
) -> Tuple[List[Dict], Optional[str]]:
    """
    Returns:
        (segments, detected_language)

    segments format:
        [{"start": float, "end": float, "text": str}, ...]
    """
    WhisperModel = _load_faster_whisper()
    if WhisperModel is None:
        raise RuntimeError("faster-whisper not installed.")

    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    segments_iter, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,
    )

    segments: List[Dict] = []
    for seg in segments_iter:
        seg_text = (getattr(seg, "text", "") or "").strip()
        if not seg_text:
            continue
        segments.append({
            "start": float(getattr(seg, "start", 0.0)),
            "end": float(getattr(seg, "end", 0.0)),
            "text": seg_text
        })

    detected_language = getattr(info, "language", None)
    return segments, detected_language


# -----------------------------
# Transcription: openai-whisper (fallback)
# -----------------------------

def _transcribe_with_openai_whisper(
    audio_path: str,
    model_name: str = "base",
    language: Optional[str] = None
) -> Tuple[List[Dict], Optional[str]]:
    """
    Returns:
        (segments, detected_language)
    """
    whisper = _load_openai_whisper()
    if whisper is None:
        raise RuntimeError("openai-whisper not installed.")

    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, language=language)

    segments: List[Dict] = []
    for seg in result.get("segments", []):
        seg_text = (seg.get("text") or "").strip()
        if not seg_text:
            continue
        segments.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": seg_text
        })

    detected_language = result.get("language")
    return segments, detected_language


# -----------------------------
# Public API: transcribe -> AudioChunk list
# -----------------------------

def transcribe_audio_file(
    audio_path: str,
    model_name: str = "base",
    language: Optional[str] = None,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    prefer_faster_whisper: bool = True
) -> List[AudioChunk]:
    """
    Transcribe a single audio file and return embedding-ready chunks.
    """
    audio_path = str(audio_path)
    source_file = Path(audio_path).name

    segments: List[Dict] = []
    detected_language: Optional[str] = None
    backend_errors: List[str] = []

    if prefer_faster_whisper:
        try:
            segments, detected_language = _transcribe_with_faster_whisper(
                audio_path=audio_path,
                model_name=model_name,
                device="cpu",
                compute_type="int8",
                language=language
            )
        except Exception as e:
            backend_errors.append(f"faster-whisper failed: {e}")

    if not segments:
        try:
            segments, detected_language = _transcribe_with_openai_whisper(
                audio_path=audio_path,
                model_name=model_name,
                language=language
            )
        except Exception as e:
            backend_errors.append(f"openai-whisper failed: {e}")

    if not segments:
        msg = "Could not transcribe audio. " + " | ".join(backend_errors) if backend_errors else \
              "Could not transcribe audio; no backend available."
        raise RuntimeError(msg)

    chunks: List[AudioChunk] = []
    chunk_index_global = 0

    for seg_idx, seg in enumerate(segments):
        seg_text = (seg.get("text") or "").strip()
        if not seg_text:
            continue

        subchunks = _chunk_text_by_chars(
            seg_text,
            max_chars=max_chars_per_chunk,
            overlap=overlap_chars
        )

        for sub in subchunks:
            sub = sub.strip()
            if not sub:
                continue

            chunk = AudioChunk(
                text=sub,
                source_file=source_file,
                chunk_index=chunk_index_global,
                char_count=len(sub),
                modality="audio",
                source_path=audio_path,
                start_sec=float(seg.get("start", 0.0)),
                end_sec=float(seg.get("end", 0.0)),
                segment_index=seg_idx,
                language=language or detected_language
            )
            chunks.append(chunk)
            chunk_index_global += 1

    return chunks


def ingest_audio_directory(
    data_dir: str = DEFAULT_DATA_DIR,
    model_name: str = "base",
    language: Optional[str] = None,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    prefer_faster_whisper: bool = True
) -> List[Dict]:
    """
    Transcribe all audio files under data_dir and return a list of dict chunks.
    """
    audio_files = find_audio_files(data_dir)
    all_chunks: List[Dict] = []

    for path in audio_files:
        try:
            chunks = transcribe_audio_file(
                audio_path=path,
                model_name=model_name,
                language=language,
                max_chars_per_chunk=max_chars_per_chunk,
                overlap_chars=overlap_chars,
                prefer_faster_whisper=prefer_faster_whisper
            )
            all_chunks.extend([c.to_dict() for c in chunks])
        except Exception as e:
            print(f"[load_audio] Skipping {path} due to error: {e}")

    return all_chunks


# -----------------------------
# for source ingestion 
# -----------------------------

def process_audio_file(audio_path: str) -> Dict:
    """
    Adapter for ingest_sources.py.

    ingest_sources expects:
      {
        "content": str,
        "modality": "audio",
        "source_file": str,
        "metadata": dict
      }
    """
    audio_path = str(audio_path)
    source_file = Path(audio_path).name

    chunks = transcribe_audio_file(audio_path)

    # Combine chunk texts into a single content string for your pipeline's raw/*.txt output.
    content = "\n\n".join([c.text for c in chunks]).strip()

    return {
        "content": content,
        "modality": "audio",
        "source_file": source_file,
        "metadata": {
            "file_size": os.path.getsize(audio_path) if os.path.exists(audio_path) else None,
            "chunk_count": len(chunks),
            "max_chars_per_chunk": DEFAULT_MAX_CHARS_PER_CHUNK,
            "overlap_chars": DEFAULT_OVERLAP_CHARS,
            "language": chunks[0].language if chunks else None,
            "backend": "faster-whisper (preferred), openai-whisper (fallback)",
        }
    }


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIR
    audio_files = find_audio_files(root)

    print(f"[load_audio] Found {len(audio_files)} audio files under '{root}'.")
    if audio_files:
        ex_path = audio_files[0]
        print(f"[load_audio] Transcribing example: {ex_path}")

        out = process_audio_file(ex_path)
        preview = (out["content"][:200] + "...") if len(out["content"]) > 200 else out["content"]

        print({
            "source_file": out["source_file"],
            "modality": out["modality"],
            "chunk_count": out["metadata"].get("chunk_count"),
            "language": out["metadata"].get("language"),
            "preview": preview
        })

