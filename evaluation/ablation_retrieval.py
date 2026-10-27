"""T8: ablation of the memory system, retrieval only. NO Groq calls (embeddings are local).

Uses what the adaptive ingest already saved (evaluation/results/eval_memory.db and the
<conv>__ingest.jsonl logs), so run run_eval.py's adaptive ingest first for each conversation.

For every answerable question whose evidence lies inside the checkpoint it checks whether
ALL evidence turns are in the top k retrieved memories (all@k), or at least one (any@k).

Variants
  full system            stored memories, 0.7 * similarity + 0.3 * score, similarity cutoff 0.15
  learned importance     same memories, ranked by similarity + 0.15 * learned score
  no importance score    similarity only (cutoff kept)
  no similarity cutoff   full system without the 0.15 cutoff
  no contradiction       also keeps the memories that were replaced by a newer one
  store everything       no small talk filter and no replacing: every turn is a memory

    python evaluation/ablation_retrieval.py --convs conv-26 conv-30 --max-turns 100
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

import locomo_utils as lu  # noqa: E402
from memory_engine import retrieve  # noqa: E402
from memory_engine.embeddings import embed  # noqa: E402
from memory_engine.importance_model import TEXT_FEATURES, text_features  # noqa: E402
from memory_engine.score import score_importance  # noqa: E402
from memory_engine.store import MemoryStore  # noqa: E402
from train_importance import make_models  # noqa: E402

KS = (5, 10)
PREFIX = re.compile(r"^\[[^\]]*\]\s*[^:]+:\s*")
W_LEARNED = 0.15


def raw(text):
    return PREFIX.sub("", text, count=1)


def learned_model_without(conv):
    t = pd.read_csv(ROOT / "evaluation" / "results" / "importance_table.csv")
    t = t[t.conv != conv]
    return make_models()["Gradient Boosting"].fit(
        t[TEXT_FEATURES].to_numpy(dtype=float), t["needed"].to_numpy())


def rank(query, items, mode, model=None):
    q = embed(query)
    sims = np.array([float(np.dot(q, it["embedding"])) for it in items])
    if mode == "learned":
        imp = model.predict_proba(np.array([text_features(raw(it["text"])) for it in items]))[:, 1]
        key = sims + W_LEARNED * imp
    elif mode == "similarity":
        key = sims
    else:
        key = 0.7 * sims + 0.3 * np.array([it["score"] for it in items])
    keep = sims >= retrieve.MIN_SIMILARITY if mode != "no_cutoff" else np.ones(len(items), bool)
    order = [i for i in np.argsort(-key) if keep[i]]
    return [items[i]["text"] for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "locomo10.json"))
    ap.add_argument("--convs", nargs="+", default=["conv-26", "conv-30"])
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--db", default=str(ROOT / "evaluation" / "results" / "eval_memory.db"))
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "results" / "ablation_retrieval.csv"))
    args = ap.parse_args()

    entries = {e["sample_id"]: e for e in lu.load_locomo(args.data)}
    store = MemoryStore(db_path=args.db)
    variants = ["full system", "learned importance", "no importance score",
                "no similarity cutoff", "no contradiction handling", "store everything"]
    hits = {v: {f"{m}@{k}": [] for m in ("all", "any") for k in KS} for v in variants}
    stored_hits = {v: [] for v in variants}
    sizes = {v: [] for v in variants}

    for conv in args.convs:
        turns = lu.conversation_turns(entries[conv])[: args.max_turns]
        by_id = {t["dia_id"]: t for t in turns}
        items = store.get_all(f"eval-{conv}")
        if not items:
            print(f"{conv}: no stored memories, skipped (run the adaptive ingest first)")
            continue
        log_path = Path(args.db).parent / f"{conv}__ingest.jsonl"
        log = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        in_db = {i["text"] for i in items}
        log_ids = {r["dia_id"] for r in log if r["action"] in ("added", "replaced")}
        gone = [t for t in turns if t["dia_id"] in log_ids and lu.format_turn(t) not in in_db]
        no_contra = items + [{"text": lu.format_turn(t), "embedding": embed(lu.format_turn(t)),
                              "score": score_importance(lu.format_turn(t), "fact")} for t in gone]
        everything = [{"text": lu.format_turn(t), "embedding": embed(lu.format_turn(t)),
                       "score": score_importance(lu.format_turn(t), "fact")} for t in turns]
        model = learned_model_without(conv)
        sets = {"full system": (items, "full"), "learned importance": (items, "learned"),
                "no importance score": (items, "similarity"), "no similarity cutoff": (items, "no_cutoff"),
                "no contradiction handling": (no_contra, "full"), "store everything": (everything, "full")}
        for v, (pool, _) in sets.items():
            sizes[v].append(len(pool))
        qs = lu.questions_within(lu.answerable_questions(entries[conv]), turns)
        n = 0
        for q in qs:
            ev = lu.evidence_texts(by_id, q["evidence"])
            if not ev:
                continue
            n += 1
            for v, (pool, mode) in sets.items():
                ranked = rank(q["question"], pool, mode, model)
                stored_hits[v].append(all(t in {p["text"] for p in pool} for t in ev))
                for k in KS:
                    top = set(ranked[:k])
                    hits[v][f"all@{k}"].append(all(t in top for t in ev))
                    hits[v][f"any@{k}"].append(any(t in top for t in ev))
        print(f"{conv}: {len(items)} stored memories, {len(gone)} replaced away, {n} questions")

    rows = []
    for v in variants:
        if not stored_hits[v]:
            continue
        rows.append({"variant": v, "memories": int(np.sum(sizes[v])), "questions": len(stored_hits[v]),
                     "evidence_stored": round(float(np.mean(stored_hits[v])), 3),
                     **{m: round(float(np.mean(x)), 3) for m, x in hits[v].items()}})
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print("\n" + out.to_string(index=False))
    print("\nmemories = total stored memories over the conversations. Small test: treat differences of a few points as noise.")


if __name__ == "__main__":
    main()