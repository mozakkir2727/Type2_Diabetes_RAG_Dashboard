import json
import random
import re

import config


def read_raw_documents():
    """Step 1: read every .txt file in data/raw/, return a list of (filename, text)."""
    documents = []
    for path in sorted(config.RAW_DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        documents.append((path.name, text))
    return documents


def split_into_chunks(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start += chunk_size - overlap
    return chunks


def load_pii_seeds():
    """Loads our list of fake (fully synthetic) patients."""
    with open(config.PII_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["synthetic_patients"]


def redact_pii(text, pii_seeds):
   
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[REDACTED_EMAIL]", text)
    text = re.sub(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", "[REDACTED_PHONE]", text)
    text = re.sub(r"\bMRN-\d{5,}\b", "[REDACTED_ID]", text, flags=re.IGNORECASE)

    for patient in pii_seeds:
        text = text.replace(patient["name"], "[REDACTED_NAME]")

    return text


def build_dataset(redact=True, seed=42, save=True):
    
    random.seed(seed)  # same random choices every run, so results are repeatable
    pii_seeds = load_pii_seeds()
    documents = read_raw_documents()

    all_chunks = []
    for filename, text in documents:
        pieces = split_into_chunks(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

        for piece in pieces:
            has_pii = False

            # Step 3: randomly decide whether this chunk gets a fake PII note
            if random.random() < config.PII_INJECTION_RATE:
                patient = random.choice(pii_seeds)
                piece = piece + "\n\n" + patient["note_sentence"]
                has_pii = True

            # Step 4: redact now, if we're building the protected version
            if has_pii and redact:
                stored_text = redact_pii(piece, pii_seeds)
            else:
                stored_text = piece

            all_chunks.append({
                "source": filename,
                "text": stored_text,
                "has_pii": has_pii,
            })

    # Step 5: save to disk (unless the caller only needs the chunks in memory)
    if save:
        with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2)

    return all_chunks


if __name__ == "__main__":
    chunks = build_dataset(redact=True)
    n_pii = sum(1 for c in chunks if c["has_pii"])
    print(f"Built {len(chunks)} chunks from {config.RAW_DATA_DIR}")
    print(f"{n_pii} of them contained fake PII, and it has been redacted.")
    print(f"Saved to {config.CHUNKS_FILE}")
