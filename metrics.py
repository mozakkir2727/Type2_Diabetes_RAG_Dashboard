import math
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_WORD = re.compile(r"[a-z0-9]+")
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _light_stem(word):
    """Very small suffix trimmer so 'prevent' matches 'prevention'/'prevents'."""
    for suffix in ("ations", "ation", "ings", "ing", "ions", "ion", "ies", "es", "s", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def content_words(text):
    """Lower-cased, stop-word-free, lightly stemmed set of words in `text`."""
    words = _WORD.findall(text.lower())
    return {_light_stem(w) for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2}


def strip_reasoning(text):
    """Removes <think>...</think> blocks some reasoning models emit."""
    return _THINK_BLOCK.sub("", text).strip()


def estimate_tokens(text):
    """Rough token estimate (~4 characters per token). Used only when the
    API doesn't report real usage numbers."""
    return math.ceil(len(text) / 4) if text else 0


def answer_relevance(question, answer):
    """Percent (0-100) of the question's content words that appear in the answer."""
    q_words = content_words(question)
    if not q_words:
        return None
    a_words = content_words(strip_reasoning(answer))
    return round(len(q_words & a_words) / len(q_words) * 100, 1)


def faithfulness(answer, context_texts, support_threshold=0.7):
    """
    Percent (0-100) of answer sentences supported by the context: a sentence
    counts as supported when at least `support_threshold` of its content
    words appear somewhere in the retrieved context. Sentences with fewer
    than 3 content words (e.g. "Sure.") are skipped. Returns None if there
    is nothing to score.
    """
    context_words = set()
    for text in context_texts:
        context_words |= content_words(text)

    checked = supported = 0
    for sentence in _SENTENCE_SPLIT.split(strip_reasoning(answer)):
        words = content_words(sentence)
        if len(words) < 3:
            continue
        checked += 1
        if len(words & context_words) / len(words) >= support_threshold:
            supported += 1
    return round(supported / checked * 100, 1) if checked else None
