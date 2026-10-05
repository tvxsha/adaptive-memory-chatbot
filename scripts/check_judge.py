"""Checks what the REAL Groq contradiction judge decides for every pair in
check_similarity.py, and whether the resulting pipeline action is right.

Pipeline action: "contradicts" or "updates"  -> REPLACE the old memory
                 "consistent" or "unrelated" -> ADD (keep both)

Two kinds of mistakes matter:
  - missed update : a real update was judged consistent/unrelated (stale fact stays)
  - wrong replace : a non-conflict was judged contradicts/updates (a correct memory is DELETED)
The second is worse.

Run from the project root with the venv activated (uses your GROQ_API_KEY):
    python scripts/check_judge.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_engine.contradiction import check_relation
from check_similarity import PAIRS  # same pairs, one source of truth

SHOULD_REPLACE = {"update", "contradiction"}


def main() -> int:
    ok = 0
    missed = []
    wrong_replace = []

    print(f"{'kind':<14} {'judge says':<12} {'action':<8} verdict")
    print("-" * 100)
    for kind, old, new in PAIRS:
        result = check_relation(old, new)
        relation = result["relation"]
        action = "replace" if relation in ("contradicts", "updates") else "add"
        want = "replace" if kind in SHOULD_REPLACE else "add"

        if action == want:
            verdict = "ok"
            ok += 1
        elif action == "replace":
            verdict = "WRONG: would DELETE a correct memory"
            wrong_replace.append((old, new, result["reasoning"]))
        else:
            verdict = "WRONG: missed update"
            missed.append((old, new, result["reasoning"]))

        print(f"{kind:<14} {relation:<12} {action:<8} {verdict}")
        print(f"{'':<14} {old}  ->  {new}")
        if result["reasoning"]:
            print(f"{'':<14} judge: {result['reasoning']}")
        elif relation == "unrelated":
            print(f"{'':<14} (no reasoning returned: if you see many of these, check the terminal "
                  f"for '[contradiction] defaulted to' lines, which mean the Groq call failed)")
        time.sleep(2.1)  # free tier allows 30 requests per minute

    print("\n=== Summary ===")
    print(f"correct actions : {ok} of {len(PAIRS)}")
    print(f"wrong replaces  : {len(wrong_replace)}   (correct memory would be deleted)")
    for old, new, why in wrong_replace:
        print(f"   {old}  ->  {new}   [{why}]")
    print(f"missed updates  : {len(missed)}")
    for old, new, why in missed:
        print(f"   {old}  ->  {new}   [{why}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())