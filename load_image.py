# load_img.py
# Wikipedia-style screenshot OCR loader with:
# - slicing for tall images
# - lightweight cleanup / dedupe
# - caching so you don't redo OCR if already done
# - optional quick-check to skip OCR if image already "has text"

import os
import re
import json
import hashlib
from difflib import SequenceMatcher
from typing import Dict, Any, List, Tuple, Optional

from PIL import Image, ImageOps
import pytesseract


# -----------------------------
# Tesseract setup
# -----------------------------
TESSERACT_CMD = os.getenv("TESSERACT_CMD") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if not os.path.exists(TESSERACT_CMD):
    raise RuntimeError(
        f"Tesseract not found at: {TESSERACT_CMD}\n"
        "Install Tesseract OR set env var TESSERACT_CMD to the full path of tesseract.exe."
    )
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
DEFAULT_LANG = os.getenv("TESSERACT_LANG", "eng")


# -----------------------------
# Helpers
# -----------------------------
def _preprocess_image(img: Image.Image) -> Image.Image:
    """Preprocess tuned for Wikipedia light-mode screenshots."""
    img = ImageOps.exif_transpose(img)
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    return img


def _safe_float(x, default=-1.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _bbox_union(bboxes: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    if not bboxes:
        return None
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return (x1, y1, x2, y2)


def _slice_horizontally(
    img: Image.Image, max_slice_height: int = 1400, overlap: int = 120
) -> List[Tuple[int, Image.Image]]:
    """Slice tall pages to improve OCR reading order."""
    w, h = img.size
    if h <= max_slice_height:
        return [(0, img)]

    slices: List[Tuple[int, Image.Image]] = []
    y = 0
    while y < h:
        y2 = min(h, y + max_slice_height)
        crop = img.crop((0, y, w, y2))
        slices.append((y, crop))
        if y2 == h:
            break
        y = y2 - overlap
    return slices


def _dedupe_lines(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove exact/near-exact duplicates (common due to slice overlap)."""
    seen = set()
    out = []
    for ln in lines:
        key = " ".join((ln.get("text") or "").lower().split())
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(ln)
    return out


def _hash_file(path: str, block_size: int = 1024 * 1024) -> str:
    """Fast enough for images; used for cache key."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _cache_path_for(image_path: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    # include file content hash to avoid stale cache if file changes
    digest = _hash_file(image_path)
    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(cache_dir, f"{base}.{digest[:16]}.ocr.json")


def _read_cache(cache_path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_cache(cache_path: str, payload: Dict[str, Any]) -> None:
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, cache_path)


def _quick_text_presence_check(
    img: Image.Image,
    lang: str,
    config: str,
    min_words: int = 8,
    min_alpha_ratio: float = 0.55,
) -> bool:
    """
    Quick/cheap check: OCR a small top-ish crop and see if we get "real" text.
    Not perfect; caching is the reliable "already done" mechanism.
    """
    w, h = img.size

    # sample: top 900px (or 20% height if smaller), minus left nav-ish margin a bit
    crop_h = min(900, max(300, int(h * 0.2)))
    left = int(w * 0.10)
    right = int(w * 0.98)
    top = 0
    bottom = crop_h

    sample = img.crop((left, top, right, bottom))
    txt = pytesseract.image_to_string(sample, lang=lang, config=config) or ""
    # Normalize
    txt = " ".join(txt.split())

    if not txt:
        return False

    words = [w for w in re.split(r"\s+", txt) if w]
    if len(words) < min_words:
        return False

    letters = sum(ch.isalpha() for ch in txt)
    if letters / max(len(txt), 1) < min_alpha_ratio:
        return False

    return True


# -----------------------------
# OCR cleaning (wiki-ish)
# -----------------------------
WIKI_NOISE_PATTERNS = [
    r"\bcreate account\b",
    r"\blog in\b",
    r"\bsearch wikipedia\b",
    r"\bmain page\b",
    r"\bcontents\b",
    r"\bcurrent events\b",
    r"\brandom article\b",
    r"\bdonate\b",
    r"\babout wikipedia\b",
    r"\bdisclaimers\b",
    r"\bprivacy policy\b",
    r"\bcookie statement\b",
    r"\bdevelopers\b",
    r"\btext available under\b",
    r"\bcreative commons\b",
    r"\bwikimedia foundation\b",
    r"\blanguages\b",
    r"\bwikidata\b",
    r"\bwikispecies\b",
    r"\bthis page was last edited\b",
    r"\bedit links\b",
    r"\bwikipedia\b",
    r"\bcreate\b.*\baccount\b",  
    r"\blog\b.*\bin\b",        
    # top / general wiki UI
    r"\bwikipedia\b",
    r"\bmain page\b",
    r"\bfeatured content\b",
    r"\bcurrent events\b",
    r"\brandom article\b",
    r"\bdonate\b",
    r"\bcreate\b.*\baccount\b",
    r"\blog\b.*\bin\b",
    r"\bedit\b",
    r"\bview history\b",
    r"\bread\b",

    # left sidebar / tools
    r"\bwhat links here\b",
    r"\brelated changes\b",
    r"\bpage information\b",
    r"\bcite this page\b",
    r"\bdownload\b.*\bqr\b",
    r"\bprintable version\b",
    r"\bwikimedia commons\b",
    r"\bupload file\b",

    # language menu (common ones; OCR often mangles so keep broad)
    r"\benglish\b|\bdeutsch\b|\bespa[nñ]ol\b|\bfran[cç]ais\b|\bitaliano\b|\bportugu[eê]s\b|\brussk?i\b|\bpolski\b|\bnederlands\b",

    # “This needs additional citations” banners
    r"\bthis needs additional citations\b",
    r"\bunsourced material\b",
    r"\blearn how and when to remove\b",



]


def _normalize_line(s: str) -> str:
    return " ".join(s.strip().split())


def _looks_like_wiki_noise(s: str) -> bool:
    for pat in WIKI_NOISE_PATTERNS:
        if re.search(pat, s, flags=re.IGNORECASE):
            return True
    return False


def _is_mostly_garbage(s: str) -> bool:
    if len(s) < 4:
        return True
    letters = sum(ch.isalpha() for ch in s)
    # tuned a bit stricter than your earlier one
    if letters / max(len(s), 1) < 0.45:
        return True
    return False


def _dedupe_similar(lines: List[str], threshold: float = 0.92) -> List[str]:
    kept: List[str] = []
    for s in lines:
        if not kept:
            kept.append(s)
            continue
        window = kept[-6:]
        if any(SequenceMatcher(None, s, prev).ratio() >= threshold for prev in window):
            continue
        kept.append(s)
    return kept

def _is_in_main_column(bbox, img_width: int) -> bool:
    """
    Heuristic for Wikipedia desktop screenshots:
    - left nav lives on far left
    - infobox / right rail lives on far right
    Keep only the middle/main content column.
    """
    x1, y1, x2, y2 = bbox

    # tuneable thresholds
    left_cut = int(img_width * 0.16)    # drop left sidebar
    right_cut = int(img_width * 0.88)   # drop right infobox/rail

    # keep if it overlaps the main column
    return (x2 >= left_cut) and (x1 <= right_cut)


def _clean_ocr_lines(raw_lines: List[str]) -> List[str]:
    out: List[str] = []
    for s in raw_lines:
        s = _normalize_line(s)
        # Drop table-of-contents style numbering lines
        if re.match(r"^\d+(\.\d+){1,6}\s+\w+", s):
            continue
        if not s:
            continue
        if _looks_like_wiki_noise(s):
            continue
        if _is_mostly_garbage(s):
            continue
        out.append(s)
    return _dedupe_similar(out)


# -----------------------------
# Main function
# -----------------------------
def process_image_file(
    image_path: str,
    lang: str = DEFAULT_LANG,
    min_line_conf: float = 30.0,
    max_lines_in_metadata: int = 80,
    slice_height: int = 1400,
    slice_overlap: int = 120,
    *,
    clean_text: bool = True,
    # "Already done" controls:
    skip_if_cached: bool = True,
    cache_dir: str = ".ocr_cache",
    skip_if_text_present: bool = False,
) -> Dict[str, Any]:
    """
    Returns common schema:
      {
        "content": str,
        "modality": "image",
        "source_file": str,
        "metadata": dict
      }

    - If skip_if_cached=True and cached result exists, returns it immediately.
    - If skip_if_text_present=True, does a quick OCR sample check and (if texty)
      returns a minimal response (without full OCR).
      For real "don't redo", rely on caching.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Cache check (reliable "already done")
    cache_path = _cache_path_for(image_path, cache_dir)
    if skip_if_cached and os.path.exists(cache_path):
        cached = _read_cache(cache_path)
        if cached and isinstance(cached, dict) and "content" in cached and "metadata" in cached:
            return cached

    img = Image.open(image_path)
    img = _preprocess_image(img)
    width, height = img.size

    config = "--oem 3 --psm 6"

    # Optional quick "does it have text already?" check (cheap heuristic)
    if skip_if_text_present:
        try:
            has_text = _quick_text_presence_check(img, lang=lang, config=config)
        except Exception:
            has_text = False

        if has_text:
            minimal = {
                "content": "",
                "modality": "image",
                "source_file": os.path.basename(image_path),
                "metadata": {
                    "path": image_path,
                    "file_size": os.path.getsize(image_path),
                    "image_width": width,
                    "image_height": height,
                    "ocr_engine": "tesseract",
                    "ocr_lang": lang,
                    "tesseract_cmd": TESSERACT_CMD,
                    "config": config,
                    "note": "Skipped full OCR because quick-check detected existing text.",
                    "skipped_ocr": True,
                },
            }
            # do NOT write this to cache; it isn't a real OCR result
            return minimal

    slices = _slice_horizontally(img, max_slice_height=slice_height, overlap=slice_overlap)

    all_lines: List[Dict[str, Any]] = []
    for y_offset, part in slices:
        data = pytesseract.image_to_data(
            part, lang=lang, config=config, output_type=pytesseract.Output.DICT
        )

        groups: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}
        n = len(data.get("text", []))
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue

            key = (
                int(data.get("block_num", [0])[i]),
                int(data.get("par_num", [0])[i]),
                int(data.get("line_num", [0])[i]),
            )
            groups.setdefault(key, []).append(
                {
                    "text": text,
                    "conf": _safe_float(data.get("conf", ["-1"])[i], -1.0),
                    "left": int(data["left"][i]),
                    "top": int(data["top"][i]) + y_offset,
                    "width": int(data["width"][i]),
                    "height": int(data["height"][i]),
                }
            )

        for key in sorted(groups.keys()):
            words = sorted(groups[key], key=lambda w: (w["top"], w["left"]))
            line_text = " ".join(w["text"] for w in words).strip()
            if not line_text:
                continue

            x1 = min(w["left"] for w in words)
            y1 = min(w["top"] for w in words)
            x2 = max(w["left"] + w["width"] for w in words)
            y2 = max(w["top"] + w["height"] for w in words)

            confs = [w["conf"] for w in words if w["conf"] >= 0]
            avg_conf = sum(confs) / len(confs) if confs else -1.0

            all_lines.append(
                {
                    "text": line_text,
                    "bbox": (x1, y1, x2, y2),
                    "avg_conf": float(avg_conf),
                    "word_count": len(words),
                }
            )

    # Confidence filter (fallback to all if nothing passes)
    good_lines = [ln for ln in all_lines if ln["avg_conf"] >= min_line_conf]
    use_lines = good_lines if good_lines else all_lines
    use_lines = [ln for ln in use_lines if _is_in_main_column(ln["bbox"], width)]

    # Dedupe from overlap
    use_lines = _dedupe_lines(use_lines)

    # Build content
    raw_text_lines = [ln["text"] for ln in use_lines]
    if clean_text:
        cleaned = _clean_ocr_lines(raw_text_lines)
        content = "\n".join(cleaned).strip()
    else:
        content = "\n".join(raw_text_lines).strip()

    overall_bbox = _bbox_union([ln["bbox"] for ln in use_lines])

    # Compact regions (bbox + conf only)
    regions = [{"bbox": ln["bbox"], "avg_conf": ln["avg_conf"]} for ln in use_lines[:max_lines_in_metadata]]

    payload = {
        "content": content,
        "modality": "image",
        "source_file": os.path.basename(image_path),
        "metadata": {
            "path": image_path,
            "file_size": os.path.getsize(image_path),
            "image_width": width,
            "image_height": height,
            "ocr_engine": "tesseract",
            "ocr_lang": lang,
            "tesseract_cmd": TESSERACT_CMD,
            "config": config,
            "min_line_conf": min_line_conf,
            "line_count": len(use_lines),
            "overall_bbox": overall_bbox,
            "regions": regions,
            "slicing": {
                "enabled": len(slices) > 1,
                "slice_height": slice_height,
                "slice_overlap": slice_overlap,
                "num_slices": len(slices),
            },
            "note": "OCR text extracted from screenshot; regions include bbox + confidence.",
            "skipped_ocr": False,
            "cache_path": cache_path,
        },
    }

    # Write cache so next time we skip work
    if skip_if_cached:
        _write_cache(cache_path, payload)

    return payload


# -----------------------------
# Convenience wrapper (optional)
# -----------------------------
def load_image(image_path: str) -> Dict[str, Any]:
    """
    Simple wrapper with sane defaults.
    Call this from your pipeline.
    """
    return process_image_file(
        image_path=image_path,
        skip_if_cached=True,
        cache_dir=".ocr_cache",
        clean_text=True,
        skip_if_text_present=False,  # leave False; caching is the real "already done"
    )


if __name__ == "__main__":
    # Quick manual test:
    import sys

    if len(sys.argv) < 2:
        print("Usage: python load_img.py path/to/image.png")
        raise SystemExit(1)

    res = load_image(sys.argv[1])
    print(res["content"][:2000])
