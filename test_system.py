from build_dataset import build_dataset, redact_pii, load_pii_seeds
from rag_system import build_search_index, retrieve, is_personal_question, answer_question

passed = 0
failed = 0


def check(description, condition):
    global passed, failed
    if condition:
        print(f"  PASS: {description}")
        passed += 1
    else:
        print(f"  FAIL: {description}")
        failed += 1


print("Building test dataset...")
chunks = build_dataset(redact=True)
vectorizer, matrix = build_search_index(chunks)
pii_seeds = load_pii_seeds()

print("\nRetrieval checks:")
results = retrieve("What lifestyle changes help prevent type 2 diabetes?", chunks, vectorizer, matrix)
check("finds at least one chunk for a prevention question", len(results) > 0)
check("top match comes from the prevention document",
      any("prevention" in r["source"] for r in results))

irrelevant = retrieve("capital of France best pizza toppings", chunks, vectorizer, matrix)
check("an unrelated question returns no confident matches", len(irrelevant) == 0)

print("\nPersonal-question detection checks:")
check("detects a dosing question",
      is_personal_question("How much insulin should I take today?"))
check("detects a self-diagnosis question",
      is_personal_question("Can you diagnose me with prediabetes?"))
check("does NOT flag a general question",
      not is_personal_question("What A1C range indicates prediabetes?"))

print("\nPII redaction checks:")
text = "Contact patient at jane.doe@example.com or 555-014-2231, record MRN-7734910."
cleaned = redact_pii(text, pii_seeds)
check("email gets redacted", "jane.doe@example.com" not in cleaned)
check("phone number gets redacted", "555-014-2231" not in cleaned)
check("medical record number gets redacted", "MRN-7734910" not in cleaned)

name_text = "Priya Raghunathan was counseled on diet."
check("known fake patient name gets redacted",
      "Priya Raghunathan" not in redact_pii(name_text, pii_seeds))

print("\nEnd-to-end pipeline checks:")
result = answer_question("What lifestyle changes help prevent type 2 diabetes?",
                          chunks, vectorizer, matrix, pii_seeds, use_huggingface=False)
check("normal question is not refused", not result["refused"])
check("normal question returns at least one source", len(result["sources"]) > 0)

result = answer_question("How much insulin should I take today?",
                          chunks, vectorizer, matrix, pii_seeds, use_huggingface=False)
check("personal-advice question is refused", result["refused"])

result = answer_question("What's the weather in Paris?",
                          chunks, vectorizer, matrix, pii_seeds, use_huggingface=False)
check("out-of-scope question is refused (not enough info)", result["refused"])

print(f"\n{passed} passed, {failed} failed")
