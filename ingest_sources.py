import os
import requests
from bs4 import BeautifulSoup
import PyPDF2

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

PDF_PATHS = [
    "data/raw_sources/doc1.pdf",
    "data/raw_sources/doc2.pdf",
]

WEB_URLS = [
    "https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Global_attributes",
    "https://blog.hubspot.com/website/website-development",
]


def pdf_to_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text

def web_to_text(url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        return ""  # skip this URL but keep script running

    soup = BeautifulSoup(resp.text, "html.parser")
    return soup.get_text(separator="\n")

def main():
    # PDFs → txt
    for i, pdf in enumerate(PDF_PATHS, 1):
        txt = pdf_to_text(pdf)
        out_path = os.path.join(RAW_DIR, f"doc_pdf_{i}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(txt)

    # Web pages → txt
    for i, url in enumerate(WEB_URLS, 1):
        txt = web_to_text(url)
        out_path = os.path.join(RAW_DIR, f"doc_web_{i}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(txt)

    print("Ingestion done: PDFs + web saved to data/raw/")

if __name__ == "__main__":
    main()