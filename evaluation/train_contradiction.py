"""T4 Level 1: a small local classifier that decides REPLACE or KEEP for an (old, new) pair.

Training data: MNLI (public). Premise = old statement, hypothesis = new statement.
    contradiction            -> replace
    entailment, neutral      -> keep
Features: MiniLM embeddings u (old) and v (new) as [u, v, |u - v|, u * v, cosine].
Models: Logistic Regression and Histogram Gradient Boosting. The one with the better
F1 on a held out slice of MNLI (NOT our test pairs) is saved.

Then it is tested on OUR labelled pairs, which it never saw in training:
    * alone (every pair)
    * gated like the real system (a pair below similarity 0.35 is simply kept)

    pip install datasets
    python evaluation/train_contradiction.py
    python evaluation/train_contradiction.py --test-csv evaluation/contradiction_majority.csv

--test-csv needs columns old_text,new_text,label (label: updates, contradicts, consistent
or unrelated). Without it the 60 author written pairs from contradiction_eval.py are used.
Embeddings are cached in evaluation/results/ so a second run is fast.
"""
import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from memory_engine import embeddings as em  # noqa: E402

RES = ROOT / "evaluation" / "results"
REPLACE = {"updates", "contradicts"}
GATE = 0.35          # same as contradiction.SIMILARITY_THRESHOLD


def embed_many(texts):
    """Fast batch embedding with the same MiniLM model the system uses."""
    if em.TextEmbedding is not None:
        model = em._get_local_model()
        vecs = np.array(list(model.embed(list(texts), batch_size=64)), dtype=np.float32)
        if vecs.ndim == 3:
            vecs = vecs.mean(axis=1)
        norm = np.linalg.norm(vecs, axis=1, keepdims=True)
        norm[norm == 0] = 1
        return vecs / norm
    return em.embed_batch(list(texts))


def pair_features(U, V):
    cos = np.sum(U * V, axis=1, keepdims=True)
    return np.hstack([U, V, np.abs(U - V), U * V, cos])


def load_mnli(n_per_group, n_val):
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/multi_nli")
    tr = ds["train"].to_pandas()
    va = ds["validation_matched"].to_pandas()
    tr = tr[tr.label.isin([0, 1, 2])]
    rep = tr[tr.label == 2].sample(n_per_group * 2, random_state=0)
    keep = pd.concat([tr[tr.label == 0].sample(n_per_group, random_state=0),
                      tr[tr.label == 1].sample(n_per_group, random_state=0)])
    train = pd.concat([rep, keep]).sample(frac=1, random_state=0)
    val = va[va.label.isin([0, 1, 2])].sample(n_val, random_state=0)
    for d in (train, val):
        d["is_replace"] = (d.label == 2).astype(int)
    return train, val


def cached_embeddings(name, texts):
    f = RES / f"{name}.npy"
    if f.exists():
        arr = np.load(f)
        if len(arr) == len(texts):
            return arr
    arr = embed_many(texts)
    np.save(f, arr)
    return arr


def load_test(path):
    if path:
        t = pd.read_csv(path, encoding="utf-8-sig")
        return t.rename(columns={"old_text": "old", "new_text": "new"})[["old", "new", "label"]]
    from contradiction_eval import PAIRS
    return pd.DataFrame(PAIRS, columns=["old", "new", "label"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-group", type=int, default=3000,
                    help="MNLI pairs per group: replace gets 2x this, keep gets 1x entailment + 1x neutral")
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--test-csv", default=None)
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)

    train, val = load_mnli(args.n_per_group, args.n_val)
    print(f"MNLI train {len(train)} (replace share {train.is_replace.mean():.2f}), "
          f"validation {len(val)} (replace share {val.is_replace.mean():.2f})")

    key = f"mnli_{args.n_per_group}_{args.n_val}"
    Ut = cached_embeddings(key + "_tr_u", train.premise.tolist())
    Vt = cached_embeddings(key + "_tr_v", train.hypothesis.tolist())
    Uv = cached_embeddings(key + "_va_u", val.premise.tolist())
    Vv = cached_embeddings(key + "_va_v", val.hypothesis.tolist())
    Xt, Xv = pair_features(Ut, Vt), pair_features(Uv, Vv)
    yt, yv = train.is_replace.to_numpy(), val.is_replace.to_numpy()

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0),
    }
    best_name, best_f1, best = None, -1, None
    print("\nMNLI validation (data the model never trained on)")
    for name, m in models.items():
        m.fit(Xt, yt)
        p = m.predict_proba(Xv)[:, 1]
        f1 = f1_score(yv, p >= 0.5)
        print(f"  {name:<20} AUC {roc_auc_score(yv, p):.3f}  F1 {f1:.3f}  "
              f"precision {precision_score(yv, p >= 0.5):.3f}  recall {recall_score(yv, p >= 0.5):.3f}")
        if f1 > best_f1:
            best_name, best_f1, best = name, f1, m
    print(f"  -> using {best_name}")

    # ---- our labelled pairs ----
    test = load_test(args.test_csv)
    U = embed_many(test.old.tolist())
    V = embed_many(test["new"].tolist())
    t0 = time.time()
    prob = best.predict_proba(pair_features(U, V))[:, 1]
    per_pair_ms = (time.time() - t0) / len(test) * 1000
    sim = np.sum(U * V, axis=1)
    expected = test.label.isin(REPLACE).to_numpy()

    def report(title, decision):
        tp = int(np.sum(decision & expected)); fp = int(np.sum(decision & ~expected))
        fn = int(np.sum(~decision & expected)); tn = int(np.sum(~decision & ~expected))
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"\n{title}  ({len(test)} pairs)")
        print(f"  correct replaces {tp}   WRONG replaces {fp}   missed replaces {fn}   correct keeps {tn}")
        print(f"  precision of replace {prec:.2f}   recall of replace {rec:.2f}   accuracy {(tp + tn) / len(test):.2f}")
        for lab in ("updates", "contradicts", "consistent", "unrelated"):
            m = (test.label == lab).to_numpy()
            if m.sum():
                good = decision[m] if lab in REPLACE else ~decision[m]
                print(f"    {lab:<12} n={m.sum():<3} handled correctly {good.sum()}/{m.sum()}")
        return decision

    for thr in (0.5,):
        d_alone = report(f"Classifier alone (threshold {thr})", prob >= thr)
        d_gated = report(f"Classifier after the similarity gate {GATE} (like the real system)",
                         (prob >= thr) & (sim >= GATE))
    print(f"\nSpeed: {per_pair_ms:.2f} ms per pair, 0 API calls, 0 tokens")
    print("\nThreshold sweep, gated (wrong replaces matter most):")
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        d = (prob >= thr) & (sim >= GATE)
        print(f"  {thr:.1f}  wrong replaces {int(np.sum(d & ~expected)):<3} "
              f"missed {int(np.sum(~d & expected)):<3} recall {np.sum(d & expected) / max(expected.sum(), 1):.2f}")

    out = test.copy()
    out["similarity"] = sim.round(3); out["p_replace"] = prob.round(3)
    out["expected"] = np.where(expected, "replace", "keep")
    out["classifier_gated"] = np.where(d_gated, "replace", "keep")
    out.to_csv(RES / "contradiction_classifier_pairs.csv", index=False)
    print("\nMistakes (gated):")
    for _, r in out[out.expected != out.classifier_gated].iterrows():
        kind = "WRONG REPLACE" if r.classifier_gated == "replace" else "MISSED"
        print(f"  {kind}: [{r.label} sim {r.similarity} p {r.p_replace}] {r.old!r} -> {r['new']!r}")

    mdir = ROOT / "memory_engine" / "models"
    mdir.mkdir(exist_ok=True)
    joblib.dump({"model": best, "name": best_name}, mdir / "contradiction_clf.joblib")
    print(f"\nsaved memory_engine/models/contradiction_clf.joblib ({best_name})")


if __name__ == "__main__":
    main()