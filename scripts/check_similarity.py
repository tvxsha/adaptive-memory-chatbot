"""Checks how the REAL embedding model scores pairs of statements against the
contradiction gate (SIMILARITY_THRESHOLD in memory_engine/contradiction.py).

Why this matters: a new statement is only sent to the LLM contradiction judge
if its similarity to an existing memory is at least SIMILARITY_THRESHOLD. If an
update like "I live in Delhi" -> "I just moved to Mumbai" scores below it, the
conflict is never detected and both facts get stored.

Run from the project root with the venv activated (uses your HF_TOKEN):
    python scripts/check_similarity.py

This is a sanity check on a handful of pairs, not a tuned result. The proper
threshold sweep needs the full contradiction test set.
"""
import sys
import time
from pathlib import Path

# Make `memory_engine` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_engine.contradiction import DUPLICATE_THRESHOLD, SIMILARITY_THRESHOLD
from memory_engine.embeddings import cosine_similarity, embed

# (kind, old statement, new statement)
# update / contradiction -> the gate SHOULD let these through to the LLM judge
# consistent             -> fine either way (the judge decides to keep both)
# unrelated              -> the gate SHOULD block these
PAIRS = [
    ("update",        "I live in Delhi.",                              "I just moved to Mumbai."),
    ("update",        "I live in Delhi and work as a data analyst.",   "Actually I just moved to Mumbai last week."),
    ("update",        "My sister is getting married in December.",     "My sister's wedding got postponed to next year."),
    ("update",        "I have a meeting on Monday.",                   "The meeting got moved to Wednesday."),
    ("contradiction", "My favorite food is biryani.",                  "I don't like biryani anymore."),
    ("contradiction", "I'm trying to learn Spanish.",                  "I gave up on learning Spanish."),
    ("contradiction", "I work at Infosys.",                            "I quit my job at Infosys last month."),
    ("contradiction", "I drive a Honda.",                              "I sold my car."),
    ("consistent",    "I live in Delhi.",                              "I live in Delhi near the metro station."),
    ("unrelated",     "I live in Delhi.",                              "My favorite color is blue."),
    ("unrelated",     "I'm learning Spanish.",                         "I hate cilantro."),
    ("unrelated",     "I work as a data analyst.",                     "I booked a flight to Goa."),
    ("unrelated",     "I live in Delhi.",                              "My friend lives in Pune."),
    ("unrelated",     "I work at Infosys.",                            "My brother works at Google."),
    ("unrelated",     "My favorite food is biryani.",                  "I'm cooking pasta tonight."),
    ("consistent",    "I'm learning Spanish.",                         "I practice Spanish on Duolingo every day."),
]

_cache = {}


def get_embedding(text):
    if text not in _cache:
        _cache[text] = embed(text)
        time.sleep(0.2)  # be gentle on the free HF tier
    return _cache[text]


def main() -> int:
    print(f"SIMILARITY_THRESHOLD = {SIMILARITY_THRESHOLD}   DUPLICATE_THRESHOLD = {DUPLICATE_THRESHOLD}\n")
    print(f"{'kind':<14} {'sim':>5}  {'gate':<8} old  ->  new")
    print("-" * 100)

    rows = []
    try:
        for kind, old, new in PAIRS:
            sim = cosine_similarity(get_embedding(old), get_embedding(new))
            if sim >= DUPLICATE_THRESHOLD:
                gate = "DUP-SKIP"
            elif sim >= SIMILARITY_THRESHOLD:
                gate = "pass"
            else:
                gate = "BELOW"
            rows.append((kind, old, new, sim, gate))
            print(f"{kind:<14} {sim:>5.2f}  {gate:<8} {old}  ->  {new}")
    except Exception as e:
        print(f"\nEmbedding call failed: {e}")
        print("Check HF_TOKEN in your .env and your internet connection.")
        print("If the error mentions the model loading (503), wait ~20 seconds and run it again.")
        return 1

    needs_gate = [r for r in rows if r[0] in ("update", "contradiction")]
    unrelated = [r for r in rows if r[0] == "unrelated"]

    caught = [r for r in needs_gate if r[3] >= SIMILARITY_THRESHOLD]
    missed = [r for r in needs_gate if r[3] < SIMILARITY_THRESHOLD]

    print("\n=== Summary ===")
    print(f"update/contradiction pairs that pass the gate at {SIMILARITY_THRESHOLD}: {len(caught)} of {len(needs_gate)}")
    for kind, old, new, sim, _ in missed:
        print(f"   MISSED ({sim:.2f}): {old}  ->  {new}")

    lowest_needed = min(r[3] for r in needs_gate)
    highest_unrelated = max(r[3] for r in unrelated)
    print(f"lowest update/contradiction similarity : {lowest_needed:.2f}")
    print(f"highest unrelated similarity           : {highest_unrelated:.2f}")
    if lowest_needed > highest_unrelated:
        print(f"-> On these examples, a threshold between {highest_unrelated:.2f} and {lowest_needed:.2f} "
              f"would catch every update/contradiction and block every unrelated pair.")
    else:
        print("-> Overlap: some unrelated pairs score as high as (or higher than) some real updates, so "
              "no single similarity threshold separates them cleanly. That would argue for judging "
              "against more candidates, or for using the LLM on every pair.")
    print("\nNote: a LOWER threshold catches more updates but sends more pairs to the LLM judge (more Groq calls).")
    return 0


if __name__ == "__main__":
    sys.exit(main())