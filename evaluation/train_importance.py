"""T3 Step B: train and test the importance models, leave one conversation out.

For each of the 10 conversations: train on the other 9, test on that one.
Models: Logistic Regression, Random Forest, Gradient Boosting.
Baselines: hand set specificity bonus, word count only, random.

    python evaluation/train_importance.py
    python evaluation/train_importance.py --with-embeddings   (adds MiniLM numbers)

Outputs (in evaluation/results/):
    importance_metrics.csv        one row per model, mean and std over 10 folds
    importance_roc.png            ROC curves (all folds pooled)
    importance_features.png       feature importance (Random Forest)
Also saves memory_engine/models/importance.joblib (trained on all conv except conv-26,
so conv-26 stays a clean test set for the later retrieval comparison).
"""
import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES = ROOT / "evaluation" / "results"

FEATURES = ["words", "chars", "has_image", "pos_in_session", "session_no",
            "pos_in_conv", "speaker_first", "has_digit", "cap_words",
            "has_question", "first_person", "specificity"]


def precision_at_top(y, score, frac=0.2):
    k = max(int(len(y) * frac), 1)
    top = np.argsort(-score)[:k]
    return float(np.mean(y[top]))


def metrics(y, score):
    return {"roc_auc": roc_auc_score(y, score),
            "pr_auc": average_precision_score(y, score),
            "prec_top20": precision_at_top(y, score)}


def make_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                random_state=0, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-embeddings", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(RES / "importance_table.csv")
    X = df[FEATURES].to_numpy(dtype=float)
    feat_names = list(FEATURES)
    if args.with_embeddings:
        from memory_engine.embeddings import embed_batch
        E = np.asarray(embed_batch(df["text"].tolist()))
        X = np.hstack([X, E])
        feat_names += [f"emb{i}" for i in range(E.shape[1])]
    y = df["needed"].to_numpy()
    convs = df["conv"].to_numpy()
    print(f"{len(df)} turns, base rate {y.mean():.3f}")

    names = list(make_models()) + ["Hand set bonus", "Word count only", "Random"]
    per_fold = {n: [] for n in names}
    pooled = {n: ([], []) for n in names}
    rng = np.random.default_rng(0)

    for c in sorted(set(convs)):
        te = convs == c
        tr = ~te
        scores = {}
        for name, model in make_models().items():
            model.fit(X[tr], y[tr])
            scores[name] = model.predict_proba(X[te])[:, 1]
        scores["Hand set bonus"] = df.loc[te, "specificity"].to_numpy()
        scores["Word count only"] = df.loc[te, "words"].to_numpy().astype(float)
        scores["Random"] = rng.random(te.sum())
        for n in names:
            per_fold[n].append(metrics(y[te], scores[n]))
            pooled[n][0].extend(y[te]); pooled[n][1].extend(scores[n])

    rows = []
    for n in names:
        f = pd.DataFrame(per_fold[n])
        rows.append({"model": n,
                     **{f"{m}_mean": f[m].mean() for m in f},
                     **{f"{m}_std": f[m].std() for m in f}})
    out = pd.DataFrame(rows).round(3)
    out.to_csv(RES / "importance_metrics.csv", index=False)
    print("\nLeave one conversation out, mean over 10 folds")
    print(out[["model", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "prec_top20_mean"]]
          .to_string(index=False))
    print(f"(base rate {y.mean():.3f}: precision at top 20 percent must beat this)")

    plt.figure(figsize=(5.2, 4.2))
    for n in names:
        yy, ss = np.array(pooled[n][0]), np.array(pooled[n][1])
        fpr, tpr, _ = roc_curve(yy, ss)
        plt.plot(fpr, tpr, label=n, ls="--" if n in names[3:] else "-")
    plt.plot([0, 1], [0, 1], color="grey", lw=0.5)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC, leave one conversation out"); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(RES / "importance_roc.png", dpi=200); plt.close()

    rf = make_models()["Random Forest"].fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=feat_names).sort_values()
    imp = imp.tail(15)
    plt.figure(figsize=(5.2, 4.2))
    plt.barh(imp.index, imp.values, color="#2F5597")
    plt.xlabel("Importance"); plt.ylabel("Feature")
    plt.title("Random Forest feature importance"); plt.tight_layout()
    plt.savefig(RES / "importance_features.png", dpi=200); plt.close()

    # model kept for the later comparison on conv-26: trained WITHOUT conv-26
    keep = convs != "conv-26"
    best = make_models()["Gradient Boosting"].fit(X[keep], y[keep])
    mdir = ROOT / "memory_engine" / "models"
    mdir.mkdir(exist_ok=True)
    joblib.dump({"model": best, "features": feat_names}, mdir / "importance.joblib")
    print("saved memory_engine/models/importance.joblib (trained without conv-26)")


if __name__ == "__main__":
    main()