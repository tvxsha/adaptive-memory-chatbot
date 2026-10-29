"""T5: train a local message categorizer and compare it with the language model.

Input : the three labelled sheets in evaluation/labels/  (labels_*.xlsx, sheet "Rows").
        30 rows are shared by all three people (used for agreement and as a clean test part);
        the other rows belong to one person each.
Model : MiniLM embeddings -> Logistic Regression and Random Forest (class weights balanced).
Split : TEST = the 30 shared rows (majority label) + 50 random other rows. TRAIN = the rest.
        Models are chosen by 5 fold cross validation on TRAIN only; TEST is scored once.
Baselines: always the most common category, and (with --llm) the language model categorizer
        on the same TEST messages (80 Groq calls, resumable, saved to results/).

    python evaluation/train_categorizer.py
    python evaluation/train_categorizer.py --llm
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from train_contradiction import embed_many  # noqa: E402

RES = ROOT / "evaluation" / "results"
CATS = ["fact", "preference", "goal", "decision", "completed_task", "recent_chat"]


def load_labels():
    files = sorted((ROOT / "evaluation" / "labels").glob("labels_*.xlsx"))
    if len(files) < 2:
        sys.exit("Put the labelled sheets in evaluation/labels/ (labels_Tvisha.xlsx etc.).")
    per = {}
    for f in files:
        d = pd.read_excel(f, sheet_name="Rows")
        d = d.rename(columns={"Label": "label"})
        blank = d.label.isna().sum()
        d["label"] = d.label.astype("string").str.strip()
        bad = d[~d.label.isin(CATS) & d.label.notna()]
        if len(bad):
            print(f"  {f.name}: {len(bad)} rows with an unknown label, ignored: {bad.label.unique().tolist()}")
        d = d[d.label.isin(CATS)]
        print(f"{f.name}: {len(d)} labelled rows ({blank} blank)")
        per[f.stem] = d.set_index("row_id")
    return per


def build_dataset(per):
    ids = [set(d.index) for d in per.values()]
    shared = set.intersection(*ids)
    print(f"\nshared rows labelled by everyone: {len(shared)}")
    rows, kappas = [], []
    if len(per) >= 2 and shared:
        names = list(per)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = per[names[i]].loc[sorted(shared), "label"]
                b = per[names[j]].loc[sorted(shared), "label"]
                k = cohen_kappa_score(a, b)
                kappas.append(k)
                print(f"  agreement {names[i]} vs {names[j]}: {np.mean(a.values == b.values):.2f}, Cohen's kappa {k:.2f}")
        print(f"  mean kappa {np.mean(kappas):.2f}")
    dropped = 0
    for rid in sorted(shared):
        votes = Counter(d.loc[rid, "label"] for d in per.values())
        lab, n = votes.most_common(1)[0]
        if n < 2:
            dropped += 1          # all different: no majority
            continue
        text = next(iter(per.values())).loc[rid, "text"]
        rows.append({"row_id": rid, "text": text, "label": lab, "part": "shared"})
    if dropped:
        print(f"  {dropped} shared rows had no majority and were dropped")
    for d in per.values():
        for rid, r in d.iterrows():
            if rid not in shared:
                rows.append({"row_id": rid, "text": r.text, "label": r.label, "part": "own"})
    return pd.DataFrame(rows)


def metrics(y, p):
    return accuracy_score(y, p), f1_score(y, p, average="macro", labels=CATS, zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="also run the language model categorizer on TEST (Groq calls)")
    ap.add_argument("--sleep", type=float, default=2.5)
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)

    df = build_dataset(load_labels())
    print(f"\n{len(df)} labelled messages. Label counts:\n{df.label.value_counts().to_string()}")

    own = df[df.part == "own"].sample(frac=1, random_state=0)
    test = pd.concat([df[df.part == "shared"], own.iloc[:50]]).reset_index(drop=True)
    train = own.iloc[50:].reset_index(drop=True)
    print(f"\ntrain {len(train)}   test {len(test)}")

    Xtr, Xte = embed_many(train.text.tolist()), embed_many(test.text.tolist())
    ytr, yte = train.label.to_numpy(), test.label.to_numpy()

    majority = Counter(ytr).most_common(1)[0][0]
    acc, f1 = metrics(yte, np.full(len(yte), majority))
    print(f"\nBaseline 'always {majority}': accuracy {acc:.3f}  macro F1 {f1:.3f}")

    models = {
        "Logistic Regression": LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                                random_state=0, n_jobs=-1),
    }
    folds = max(2, min(5, int(min(Counter(ytr).values()))))
    skf = StratifiedKFold(folds, shuffle=True, random_state=0)
    best_name, best_cv = None, -1
    print(f"\n{folds} fold cross validation on TRAIN")
    for name, m in models.items():
        p = cross_val_predict(m, Xtr, ytr, cv=skf)
        a, f = metrics(ytr, p)
        print(f"  {name:<20} accuracy {a:.3f}  macro F1 {f:.3f}")
        if f > best_cv:
            best_name, best_cv = name, f
    best = models[best_name].fit(Xtr, ytr)
    print(f"  -> using {best_name}")

    t0 = time.time(); pred = best.predict(Xte); ms = (time.time() - t0) / len(test) * 1000
    acc, f1 = metrics(yte, pred)
    print(f"\nTEST ({len(test)} messages, scored once): {best_name}  accuracy {acc:.3f}  macro F1 {f1:.3f}  ({ms:.2f} ms per message)")
    print(classification_report(yte, pred, labels=CATS, zero_division=0))
    cm = confusion_matrix(yte, pred, labels=CATS)

    llm_pred = None
    if args.llm:
        from memory_engine.categorize import categorize
        cache = RES / "categorizer_llm_preds.csv"
        done = pd.read_csv(cache).set_index("row_id").llm.to_dict() if cache.exists() else {}
        for _, r in test.iterrows():
            if int(r.row_id) in done:
                continue
            res = categorize(r.text)
            if res.get("error"):
                print("  language model error, stopping (run again later):", res["error"][:150])
                break
            done[int(r.row_id)] = res["category"]
            pd.DataFrame({"row_id": list(done), "llm": list(done.values())}).to_csv(cache, index=False)
            time.sleep(args.sleep)
        have = test.row_id.astype(int).isin(done)
        if have.all():
            llm_pred = test.row_id.astype(int).map(done).to_numpy()
            a, f = metrics(yte, llm_pred)
            print(f"LANGUAGE MODEL on the same TEST messages: accuracy {a:.3f}  macro F1 {f:.3f}")
            print(classification_report(yte, llm_pred, labels=CATS, zero_division=0))
        else:
            print(f"language model predictions available for {int(have.sum())}/{len(test)} so far")

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CATS))); ax.set_xticklabels(CATS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(CATS))); ax.set_yticklabels(CATS, fontsize=8)
    for i in range(len(CATS)):
        for j in range(len(CATS)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("Predicted category"); ax.set_ylabel("Human label")
    ax.set_title(f"{best_name} on {len(test)} test messages")
    fig.tight_layout(); fig.savefig(RES / "categorizer_confusion.png", dpi=200)

    out = test.assign(predicted=pred)
    if llm_pred is not None:
        out["llm"] = llm_pred
    out.to_csv(RES / "categorizer_test_predictions.csv", index=False)
    mdir = ROOT / "memory_engine" / "models"; mdir.mkdir(exist_ok=True)
    joblib.dump({"model": best, "name": best_name, "classes": CATS}, mdir / "categorizer.joblib")
    print("saved memory_engine/models/categorizer.joblib and evaluation/results/categorizer_confusion.png")


if __name__ == "__main__":
    main()