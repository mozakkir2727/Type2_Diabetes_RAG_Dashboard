import json
import re
import sys
import time

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from build_dataset import redact_pii, load_pii_seeds
from metrics import answer_relevance, faithfulness, estimate_tokens


# ---------- Step 1 & 2: load chunks, build a search index ----------

def load_chunks():
    with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_search_index(chunks):
    
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def retrieve(question, chunks, vectorizer, matrix):
    """Finds the TOP_K chunks most similar to the question."""
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, matrix)[0]

    # Get chunk indexes sorted by score, best match first
    ranked_indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for i in ranked_indexes[:config.TOP_K]:
        if scores[i] >= config.MIN_SIMILARITY:
            match = dict(chunks[i])  # copy so we don't modify the original
            match["score"] = float(scores[i])
            results.append(match)
    return results


# ---------- Step 3: safety check on the QUESTION ----------

PERSONAL_QUESTION_PATTERNS = [
    r"\bshould i\b",
    r"\bam i\b.*\bdiabet",
    r"\bdo i have\b",
    r"\bwhat dose\b",
    r"\bhow much (insulin|metformin|medication)\b",
    r"\bmy (blood sugar|a1c|glucose)\b",
    r"\bdiagnos(e|is) me\b",
    r"\bprescribe\b",
]


def is_personal_question(question):
    question_lower = question.lower()
    return any(re.search(pattern, question_lower) for pattern in PERSONAL_QUESTION_PATTERNS)


# ---------- Step 5: build the prompt and call Hugging Face ----------

def build_messages(question, context_chunks):
    
    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )
    user_message = (
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer using only the context above. If it's not enough, say so."
    )
    return [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def ask_huggingface(messages):
   
    if not config.HF_TOKEN:
        raise RuntimeError(
            "No Hugging Face token found. Get a free one at "
            "https://huggingface.co/settings/tokens and set it with:\n"
            "  export HF_API_TOKEN=hf_xxx"
        )

    headers = {"Authorization": f"Bearer {config.HF_TOKEN}"}
    payload = {
        "model": config.HF_MODEL,
        "messages": messages,
        "max_tokens": config.MAX_NEW_TOKENS,
        "temperature": config.TEMPERATURE,
    }

    print("DEBUG - sending payload:", payload)  # TEMPORARY - remove once this works

    # The chosen provider can occasionally be busy (503); retry once.
    response = None
    for attempt in range(2):
        response = requests.post(config.HF_API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 503:
            time.sleep(5)
            continue
        break

    if response.status_code != 200:
        raise RuntimeError(f"Hugging Face API error {response.status_code}: {response.text}")

    result = response.json()
    return result["choices"][0]["message"]["content"].strip(), result.get("usage")


# ---------- Step 6: put it all together ----------

def answer_question(question, chunks, vectorizer, matrix, pii_seeds,
                     use_huggingface=True, redact_output=True):

    # Safety check #1: refuse personal medical questions right away
    if is_personal_question(question):
        return {"answer": config.PERSONAL_ADVICE_REFUSAL_MSG, "sources": [], "refused": True}

    # Find the most relevant chunks for this question
    matches = retrieve(question, chunks, vectorizer, matrix)
    if not matches:
        return {"answer": config.NOT_ENOUGH_INFO_MSG, "sources": [], "refused": True}

    # Generate the answer -- either with the real model, or (for fast/offline
    # testing) just by returning the matched text directly
    messages = build_messages(question, matches)
    usage = None
    if use_huggingface:
        raw_answer, usage = ask_huggingface(messages)
    else:
        raw_answer = "\n\n".join(f"From {m['source']}: {m['text']}" for m in matches)

    # Safety check #2: redact PII from the final answer (see docstring above)
    final_answer = redact_pii(raw_answer, pii_seeds) if redact_output else raw_answer

    sources = sorted(set(m["source"] for m in matches))

    # Quality / cost numbers for the dashboard. Real token counts when the
    # API reports them, otherwise a rough estimate (flagged as such).
    if usage and usage.get("total_tokens") is not None:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage["total_tokens"]
        tokens_estimated = False
    else:
        prompt_tokens = estimate_tokens("\n".join(m["content"] for m in messages))
        completion_tokens = estimate_tokens(raw_answer)
        total_tokens = prompt_tokens + completion_tokens
        tokens_estimated = True

    return {
        "answer": final_answer,
        "sources": sources,
        "refused": False,
        "metrics": {
            "generated": use_huggingface,   # False = raw retrieved text, not a model answer
            "relevance": answer_relevance(question, final_answer),
            "faithfulness": faithfulness(final_answer, [m["text"] for m in matches]),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_estimated": tokens_estimated,
        },
    }


def main():
    question = " ".join(sys.argv[1:]) or "What is type 2 diabetes?"

    chunks = load_chunks()
    vectorizer, matrix = build_search_index(chunks)
    pii_seeds = load_pii_seeds()

    try:
        result = answer_question(question, chunks, vectorizer, matrix, pii_seeds)
    except RuntimeError as error:
        print(f"\nCouldn't get an answer: {error}")
        return

    print("\nQUESTION:", question)
    print("\nANSWER:\n", result["answer"])
    if result["sources"]:
        print("\nSOURCES:", ", ".join(result["sources"]))


if __name__ == "__main__":
    main()
