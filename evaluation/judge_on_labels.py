"""Runs the real LLM judge on independently labelled pairs and scores it.

Input CSV columns: old_text,new_text,label   (label: updates/contradicts/consistent/unrelated)
Pairs below the similarity gate are never sent to the judge (decision = keep), exactly like the live system.
Resumable: judge answers are cached in the output CSV, so a re-run only calls Groq for missing pairs.

    python evaluation/judge_on_labels.py --csv evaluation/contradiction_majority.csv
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from memory_engine import contradiction  # noqa: E402
from memory_engine.embeddings import cosine_similarity, embed  # noqa: E402

REPLACE = {"updates", "contradicts"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "evaluation" / "contradiction_majority.csv"))
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "contradiction_majority_judged.csv"))
    ap.add_argument("--sleep", type=float, default=2.5)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out = Path(args.out)
    cache = {}
    if out.exists():
        prev = pd.read_csv(out)
        for _, r in prev.iterrows():
            if isinstance(r.get("relation"), str) and r["relation"] and not r.get("error", False):
                cache[(r.old_text, r.new_text)] = r.relation
    thr = contradiction.SIMILARITY_THRESHOLD
    rows = []
    for i, r in df.iterrows():
        sim = cosine_similarity(embed(r.old_text), embed(r.new_text))
        gated = sim >= thr
        rel, err = "", False
        if gated:
            key = (r.old_text, r.new_text)
            if key in cache:
                rel = cache[key]
            else:
                res = contradiction.check_relation(r.old_text, r.new_text)
                rel, err = res["relation"], "error" in res
                time.sleep(args.sleep)
                print(f"[{i+1}/{len(df)}] {rel:12} (label {r.label})", flush=True)
        decision = "replace" if (gated and rel in REPLACE) else "keep"
        rows.append({"old_text": r.old_text, "new_text": r.new_text, "label": r.label,
                     "similarity": round(sim, 3), "gated": gated, "relation": rel,
                     "decision": decision, "error": err})
        pd.DataFrame(rows).to_csv(out, index=False)   # save as we go

    res = pd.DataFrame(rows)
    exp = res.label.isin(REPLACE)
    pred = res.decision == "replace"
    tp, fp, fn = int((pred & exp).sum()), int((pred & ~exp).sum()), int((~pred & exp).sum())
    print(f"\n{len(res)} pairs, gate {thr}: {int(res.gated.sum())} reached the judge, "
          f"{int(res.error.sum())} judge errors")
    print(f"Gate: real replaces blocked by the gate = {int((exp & ~res.gated).sum())} of {int(exp.sum())}")
    print(f"LANGUAGE MODEL JUDGE: correct replaces {tp}  WRONG replaces {fp}  missed {fn}")
    print(f"precision {tp/max(tp+fp,1):.2f}  recall {tp/max(tp+fn,1):.2f}")
    print("\nConfusion (rows = human label, columns = judge relation, gated pairs only):")
    print(pd.crosstab(res[res.gated].label, res[res.gated].relation))
    if res.error.any():
        print("\nWARNING: some judge calls failed (bad key or quota). Fix the key and re-run; it resumes.")


if __name__ == "__main__":
    main()