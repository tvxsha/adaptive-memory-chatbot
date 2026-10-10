"""Retrieval-only analysis: NO Groq calls, so it costs no quota.

For every answerable question inside the ingested checkpoint it checks where the
evidence turns end up in the memory ranking, and compares ranking strategies.
It answers: "when adaptive gets a question wrong, is it because the memory was
never stored, because it was ranked too low, or because of the model?"

Run after the adaptive ingest has finished:
    python evaluation/retrieval_analysis.py --conv conv-26 --max-turns 100
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

import locomo_utils as lu  # noqa: E402
from memory_engine import retrieve  # noqa: E402
from memory_engine.embeddings import cosine_similarity, embed  # noqa: E402
from memory_engine.score import score_importance  # noqa: E402
from memory_engine.store import MemoryStore  # noqa: E402 
import json  # noqa: E402

KS = (3, 5, 10, 20)


def rank_texts(query, items, strategy):
    q = embed(query)
    scored = []
    for it in items:
        sim = cosine_similarity(q, it["embedding"])
        if strategy == "similarity_only":
            key = sim
        elif strategy == "combined_no_cutoff":
            key = 0.7 * sim + 0.3 * it["score"]
        else:  # combined + MIN_SIMILARITY cutoff = what the system does now
            if sim < retrieve.MIN_SIMILARITY:
                continue
            key = 0.7 * sim + 0.3 * it["score"]
        scored.append((key, it["text"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "locomo10.json"))
    ap.add_argument("--conv", default="conv-26")
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--db", default=str(ROOT / "evaluation" / "results" / "eval_memory.db"))
    ap.add_argument("--keep-recent", action="store_true",
                    help="simulate STORING the turns the pipeline skipped as recent_chat filler "
                    "(free: embeddings run locally, no Groq calls)")
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "figures" / "retrieval_analysis.csv"))
    args = ap.parse_args()

    entry = {e["sample_id"]: e for e in lu.load_locomo(args.data)}[args.conv]
    turns = lu.conversation_turns(entry)[: args.max_turns]
    by_id = {t["dia_id"]: t for t in turns}
    questions = lu.questions_within(lu.answerable_questions(entry), turns)
    items = MemoryStore(db_path=args.db).get_all(f"eval-{args.conv}")
    if not items:
        raise SystemExit("No stored memories found: run the adaptive ingest first.")

    log_path = Path(args.db).parent / f"{args.conv}__ingest.jsonl"
    log = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()] if log_path.exists() else []
    if log:
        print("What happened to each ingested turn:")
        for (action, reason), c in Counter((r["action"], (r.get("reasoning") or "")[:45]) for r in log).most_common(8):
            print(f"  {c:>3}  {action:<9} {reason}")
        print()
    if args.keep_recent:
        skipped = {r["dia_id"] for r in log if r["action"] == "skip" and r.get("category") == "recent_chat"}
        added = 0
        for t in turns:
            if t["dia_id"] in skipped:
                text = lu.format_turn(t)
                items.append({"text": text, "category": "recent_chat", "embedding": embed(text),
                              "score": score_importance(text, "recent_chat")})
                added += 1
        print(f"SIMULATION: also keeping {added} turns that were skipped as recent_chat\n")
    stored = {i["text"] for i in items}

    print(f"{len(turns)} turns ingested -> {len(items)} stored memories")
    print("memories by category:", dict(Counter(i["category"] for i in items)))
    n_turn_stored = sum(1 for t in turns if lu.format_turn(t) in stored)
    print(f"turns still stored verbatim: {n_turn_stored}/{len(turns)}\n")

    rows = []
    for q in questions:
        ev = lu.evidence_texts(by_id, q["evidence"])
        if not ev:
            continue
        row = {"idx": q["idx"], "category": lu.CATEGORY_NAMES[q["category"]], "n_evidence": len(ev),
               "all_evidence_stored": all(t in stored for t in ev)}
        for strategy in ("current", "combined_no_cutoff", "similarity_only"):
            ranked = rank_texts(q["question"], items, strategy)
            for k in KS:
                row[f"{strategy}@{k}"] = all(t in set(ranked[:k]) for t in ev)
        rows.append(row)

    n = len(rows)
    print(f"{n} questions with evidence inside the checkpoint\n")
    stored_rate = sum(r["all_evidence_stored"] for r in rows) / n
    print(f"evidence still stored (never dropped or replaced): {stored_rate:.2f}")
    print(f"\n{'strategy':<22}" + "".join(f"{'all@' + str(k):>8}" for k in KS))
    for strategy in ("current", "combined_no_cutoff", "similarity_only"):
        print(f"{strategy:<22}" + "".join(f"{sum(r[f'{strategy}@{k}'] for r in rows) / n:>8.2f}" for k in KS))

    print("\nBy question type (current strategy, all evidence in top-5):")
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        print(f"  {cat:<12} n={len(sub):<3} stored={sum(r['all_evidence_stored'] for r in sub) / len(sub):.2f}"
              f"  top5={sum(r['current@5'] for r in sub) / len(sub):.2f}")

    missed = [r for r in rows if r["all_evidence_stored"] and not r["current@5"]]
    never = [r for r in rows if not r["all_evidence_stored"]]
    print(f"\nWhere the failures come from (all evidence NOT in top-5): {sum(not r['current@5'] for r in rows)}/{n}")
    print(f"  never stored / replaced: {len(never)}")
    print(f"  stored but ranked below top-5: {len(missed)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-question details: {args.out}")


if __name__ == "__main__":
    main()