import time
from collections import Counter, deque

from flask import Flask, render_template, request, jsonify

import config
from rag_system import load_chunks, build_search_index, answer_question
from build_dataset import load_pii_seeds, build_dataset
from evaluate import run_all_questions, TEST_QUESTIONS

app = Flask(__name__)

# answer noticeably slower for no reason.
print("Loading RAG system...")
if not config.CHUNKS_FILE.exists():
    # First run on a fresh deployment -- the processed dataset doesn't
    # exist yet, so build it automatically instead of crashing.
    print("No processed dataset found yet -- building it now...")
    build_dataset(redact=True)
chunks = load_chunks()
vectorizer, matrix = build_search_index(chunks)
pii_seeds = load_pii_seeds()
print("Ready. Open http://127.0.0.1:5000 in your browser.")

# It lives in memory, so it resets when the server restarts.
live_log = deque(maxlen=200)


def classify_status(result):
    if not result["refused"]:
        return "answered"
    if result["answer"] == config.PERSONAL_ADVICE_REFUSAL_MSG:
        return "personal_advice"
    if result["answer"] == config.NOT_ENOUGH_INFO_MSG:
        return "insufficient_info"
    return "refused"  # fallback label, shouldn't normally happen


@app.route("/")
def home():
    """Serves the webpage itself."""
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def ask():
    
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Please type a question."}), 400

    start = time.perf_counter()
    try:
        result = answer_question(question, chunks, vectorizer, matrix, pii_seeds)
    except RuntimeError as error:
        # e.g. missing Hugging Face token, or a network/API problem --
        # show a clear message on the page instead of a server crash
        return jsonify({"error": str(error)}), 500
    elapsed_ms = round((time.perf_counter() - start) * 1000)

    result["status"] = classify_status(result)
    result["latency_ms"] = elapsed_ms

    m = result.get("metrics")
    if not result["refused"] and m:
        live_log.append({
            "latency_ms": elapsed_ms,
            "relevance": m["relevance"],
            "faithfulness": m["faithfulness"],
            "prompt_tokens": m["prompt_tokens"],
            "completion_tokens": m["completion_tokens"],
            "total_tokens": m["total_tokens"],
            "tokens_estimated": m["tokens_estimated"],
        })
    return jsonify(result)


@app.route("/dashboard")
def dashboard():
    """Serves the performance dashboard page."""
    return render_template("dashboard.html")


@app.route("/api/performance")
def performance():
    pii_seeds = load_pii_seeds()

    protected_chunks = build_dataset(redact=True, save=False)
    protected_vectorizer, protected_matrix = build_search_index(protected_chunks)
    protected_scores = run_all_questions(
        protected_chunks, protected_vectorizer, protected_matrix, pii_seeds,
        use_huggingface=False, redact_output=True,
    )

    vulnerable_chunks = build_dataset(redact=False, save=False)
    vulnerable_vectorizer, vulnerable_matrix = build_search_index(vulnerable_chunks)
    vulnerable_scores = run_all_questions(
        vulnerable_chunks, vulnerable_vectorizer, vulnerable_matrix, pii_seeds,
        use_huggingface=False, redact_output=False,
    )

    return jsonify({"protected": protected_scores, "vulnerable": vulnerable_scores})


def live_summary():
    """Averages over the real questions answered so far (see live_log)."""
    entries = list(live_log)

    def avg(key):
        values = [e[key] for e in entries if e[key] is not None]
        return round(sum(values) / len(values), 1) if values else None

    latencies = sorted(e["latency_ms"] for e in entries)
    return {
        "sample_count": len(entries),
        "avg_latency_ms": avg("latency_ms"),
        "p95_latency_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else None,
        "avg_relevance": avg("relevance"),
        "avg_faithfulness": avg("faithfulness"),
        "avg_prompt_tokens": avg("prompt_tokens"),
        "avg_completion_tokens": avg("completion_tokens"),
        "avg_total_tokens": avg("total_tokens"),
        "total_tokens": sum(e["total_tokens"] for e in entries),
        "any_tokens_estimated": any(e["tokens_estimated"] for e in entries),
        "recent": entries[-8:][::-1],   # newest first
    }


@app.route("/api/dataset-stats")
def dataset_stats():
    per_source = Counter()
    pii_per_source = Counter()
    total_chars = 0

    for c in chunks:
        per_source[c["source"]] += 1
        total_chars += len(c["text"])
        if c["has_pii"]:
            pii_per_source[c["source"]] += 1

    sources = [
        {
            "source": name,
            "chunk_count": per_source[name],
            "pii_count": pii_per_source.get(name, 0),
        }
        for name in sorted(per_source)
    ]

    question_counts = {
        "normal": len(TEST_QUESTIONS["normal"]),
        "out_of_scope": len(TEST_QUESTIONS["out_of_scope"]),
        "trying_to_get_pii": len(TEST_QUESTIONS["trying_to_get_pii"]),
        "personal_advice": len(TEST_QUESTIONS["personal_advice"]),
    }

    return jsonify({
        "total_chunks": len(chunks),
        "total_sources": len(per_source),
        "chunks_with_pii": sum(pii_per_source.values()),
        "avg_chunk_length": round(total_chars / len(chunks)) if chunks else 0,
        "sources": sources,
        "question_counts": question_counts,
        "live": live_summary(),
        "config": {
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k": config.TOP_K,
            "min_similarity": config.MIN_SIMILARITY,
            "pii_injection_rate": round(config.PII_INJECTION_RATE * 100),
            "hf_model": config.HF_MODEL,
        },
    })


if __name__ == "__main__":
    # to anyone who triggers an error on a public URL.
    import os
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
