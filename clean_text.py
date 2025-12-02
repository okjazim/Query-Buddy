import os
import string

RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"
os.makedirs(CLEAN_DIR, exist_ok=True)

def clean_text(text):
    text = text.lower()  #lowercasing
    text = text.translate(str.maketrans('', '', string.punctuation))  #removing punctuations
    text = " ".join(text.split())  # replacing multiple spaces and new lines with a single space
    return text

if __name__ == "__main__":
    for filename in os.listdir(RAW_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(RAW_DIR, filename), "r", encoding="utf-8") as infile:
                raw_text = infile.read()
            cleaned = clean_text(raw_text)
            out_name = filename.replace(".txt", "_clean.txt")
            with open(os.path.join(CLEAN_DIR, out_name), "w", encoding="utf-8") as outfile:
                outfile.write(cleaned)

    print("The cleaning process is done. Files are in data/clean/")
