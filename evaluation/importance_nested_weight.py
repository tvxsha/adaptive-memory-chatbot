"""T3 Step C2: choose the importance weight WITHOUT peeking, then test.

For each held out conversation c:
  1. choose the weight w using only the other 9 conversations
     (each of them scored by a model trained on the remaining 8, so c is never used),
  2. train on those 9, score c with that w,
  3. compare with similarity only on the same questions.
Then a paired bootstrap gives a 95 percent interval for the gain.

    python evaluation/importance_nested_weight.py
Needs evaluation/results/importance_table.csv and turn_embeddings.npy
(run build_importance_data.py and compare_importance_retrieval.py first).
Takes a few minutes (about 100 small models).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

import locomo_utils as lu  # noqa: E402
from train_importance import FEATURES, make_models  # noqa: E402

RES = ROOT / "evaluation" / "results"
GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.43)
K = 5
rng = np.random.default_rng(0)


def main():
    df = pd.read_csv(RES / "importance_table.csv")
    E = np.load(RES / "turn_embeddings.npy")
    from memory_engine.embeddings import embed
    entries = {e["sample_id"]: e for e in lu.load_locomo(str(ROOT / "data" / "locomo10.json"))}
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["needed"].to_numpy()
    convs = df["conv"].to_numpy()
    names = sorted(set(convs))

    # per conversation: question similarity vectors and evidence positions
    qdata = {}
    for c in names:
        idx = np.where(convs == c)[0]
        pos = {d: i for i, d in enumerate(df.loc[idx, "dia_id"])}
        items = []
        for q in lu.answerable_questions(entries[c]):
            ev = {pos[e] for e in q["evidence"] if e in pos}
            if ev:
                items.append((E[idx] @ embed(q["question"]), ev))
        qdata[c] = (idx, items)

    def eval_conv(c, train_mask):
        """-> array [len(GRID), n_questions, 2]  (all@K, any@K) for each weight."""
        idx, items = qdata[c]
        model = make_models()["Gradient Boosting"].fit(X[train_mask], y[train_mask])
        imp = model.predict_proba(X[idx])[:, 1]
        out = np.zeros((len(GRID), len(items), 2))
        for qi, (sim, ev) in enumerate(items):
            for wi, w in enumerate(GRID):
                top = set(np.argsort(-(sim + w * imp))[:K].tolist())
                out[wi, qi] = (ev <= top, bool(ev & top))
        return out

    base, nested, chosen, per_conv = [], [], {}, []
    for c in names:
        # step 1: pick w on the other 9, each model trained without c and without itself
        inner = []
        for j in names:
            if j == c:
                continue
            m = (convs != c) & (convs != j)
            inner.append(eval_conv(j, m))
        inner = np.concatenate(inner, axis=1)           # weights x questions x 2
        w_best = GRID[int(np.argmax(inner[:, :, 0].mean(axis=1)))]
        chosen[c] = w_best
        # step 2: train on the 9, score c
        res = eval_conv(c, convs != c)
        b = res[0]                                      # w = 0 is similarity only
        n = res[GRID.index(w_best)]
        base.append(b); nested.append(n)
        per_conv.append((c, len(b), b[:, 0].mean(), n[:, 0].mean(), w_best))
        print(f"{c}: chosen w={w_best}  all@{K} {b[:, 0].mean():.3f} -> {n[:, 0].mean():.3f}", flush=True)

    base = np.concatenate(base); nested = np.concatenate(nested)
    print(f"\nPooled over {len(base)} questions (nested weight choice, no peeking)")
    for j, m in enumerate((f"all@{K}", f"any@{K}")):
        d = nested[:, j] - base[:, j]
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"{m}: similarity only {base[:, j].mean():.3f}  learned {nested[:, j].mean():.3f}  "
              f"gain {d.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    pc = pd.DataFrame(per_conv, columns=["conv", "questions", "similarity_only", "learned", "chosen_w"]).round(3)
    pc.to_csv(RES / "importance_nested_by_conv.csv", index=False)
    wins = int((pc.learned > pc.similarity_only).sum())
    print(f"\nLearned beat similarity only in {wins} of {len(pc)} conversations")
    print(pc.to_string(index=False))


if __name__ == "__main__":
    main()