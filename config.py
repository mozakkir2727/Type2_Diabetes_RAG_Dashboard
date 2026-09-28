import os
from pathlib import Path

# --- Folders and files ---
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PII_FILE = BASE_DIR / "data" / "pii_seeds.json"
CHUNKS_FILE = BASE_DIR / "data" / "processed_chunks.json"

# --- Chunking: how we split documents into smaller pieces ---
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 120    # characters shared between one chunk and the next,
                        # so we don't cut a sentence in half at the boundary

# --- How much fake PII to mix into the dataset ---
PII_INJECTION_RATE = 0.25   # 25% of chunks 
# --- Retrieval: how we find the best matching chunks for a question ---
TOP_K = 4              # how many chunks to retrieve per question
MIN_SIMILARITY = 0.12  # if the best match is below this score, treat it as "no relevant info"
                        # (tuned by hand for this small ~10-chunk demo dataset)


# Any model listed at https://huggingface.co/docs/inference-providers can
# go here; ":fastest" tells the router to auto-pick a fast free provider.
HF_MODEL = "deepseek-ai/DeepSeek-R1:fastest"
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
  # set this in your shell, never hard-code a token here
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.01     # close to 0 = more factual, less "creative" answers

# --- Fixed messages the system uses ---
NOT_ENOUGH_INFO_MSG = (
    "I don't have enough verified information from my sources to answer that. "
    "Please consult a licensed healthcare provider or check CDC / WHO / NIH directly."
)

PERSONAL_ADVICE_REFUSAL_MSG = (
    "I can only give general public-health information about Type 2 Diabetes, "
    "not personal medical advice. Please talk to a doctor about your own situation."
)

SYSTEM_PROMPT = (
    "You are a public-health assistant that answers questions about Type 2 Diabetes "
    "using ONLY the context given to you below. Rules:\n"
    "1. Only use the context provided -- do not use outside knowledge.\n"
    "2. If the context doesn't contain the answer, say so honestly instead of guessing.\n"
    "3. Never give personal medical advice, diagnosis, or medication dosing.\n"
    "4. Never repeat any names, emails, phone numbers, or ID numbers you see in the context.\n"
    "5. Mention which source each fact came from, if you can.\n"
)
