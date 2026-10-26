"""T3 Step C: does the learned importance score rank evidence better than the hand set one?

No Groq calls. Embeddings run locally (first run embeds all 5,882 turns, about
1 to 3 minutes, then they are cached in evaluation/results/turn_embeddings.npy).

For each conversation (the importance model is trained on the OTHER 9, so it
never sees the conversation it is scored on) and each answerable question:
  rank every turn of the conversation by   similarity + w * importance
and check whether the evidence turns land in the top k.

    python evaluation/compare_importance_retrieval.py

all@k : ALL evidence turns are in the top k (strict)
any@k : AT LEAST ONE evidence turn is in the top k (lenient)
The current system uses 0.7 * similarity + 0.3 * score, which is the same ranking
as similarity + (0.3 / 0.7) * score, so w = 0.43 is the current setting.
Hand set importance here = 0.9 + specificity bonus (every turn treated as a
fact, because assigning real categories needs the language model).
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
KS = (5, 10, 20)
WEIGHTS = (0.0, 0.1, 0.2, 0.43, 0.8)


def get_embeddings(df):
    cache = RES / "turn_embeddings.npy"
    if cache.exists():
        E = np.load(cache)
        if len(E) == len(df):
            return E
    from memory_engine.embeddings import embed_batch
    print(f"embedding {len(df)} turns locally ...")
    E = np.asarray(embed_batch(df["emb_text"].tolist()), dtype=np.float32)
    np.save(cache, E)
    return E


def main():
    df = pd.read_csv(RES / "importance_table.csv")
    entries = {e["sample_id"]: e for e in lu.load_locomo(str(ROOT / "data" / "locomo10.json"))}
    # text exactly as the system stores it: [date] speaker: text
    emb_text = []
    for conv, g in df.groupby("conv", sort=False):
        turns = {t["dia_id"]: t for t in lu.conversation_turns(entries[conv])}
        emb_text += [lu.format_turn(turns[d]) for d in g["dia_id"]]
    df["emb_text"] = emb_text
    E = get_embeddings(df)
    from memory_engine.embeddings import embed
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["needed"].to_numpy()

    hits = {}   # (scorer, w) -> {metric: [bool,...]}
    for conv in sorted(df["conv"].unique()):
        te = (df["conv"] == conv).to_numpy()
        model = make_models()["Gradient Boosting"].fit(X[~te], y[~te])
        learned = model.predict_proba(X[te])[:, 1]
        hand = 0.9 + df.loc[te, "specificity"].to_numpy()
        ids = df.loc[te, "dia_id"].tolist()
        pos = {d: i for i, d in enumerate(ids)}
        Ec = E[te]
        for q in lu.answerable_questions(entries[conv]):
            ev = {pos[e] for e in q["evidence"] if e in pos}
            if not ev:
                continue
            sim = Ec @ embed(q["question"])
            for scorer, s in (("similarity only", None), ("hand set", hand), ("learned", learned)):
                for w in ((0.0,) if s is None else WEIGHTS):
                    comb = sim if s is None else sim + w * s
                    order = np.argsort(-comb)
                    rec = hits.setdefault((scorer, w), {f"{m}@{k}": [] for m in ("all", "any") for k in KS})
                    for k in KS:
                        top = set(order[:k].tolist())
                        rec[f"all@{k}"].append(ev <= top)
                        rec[f"any@{k}"].append(bool(ev & top))

    rows = []
    for (scorer, w), r in hits.items():
        rows.append({"scorer": scorer, "weight_w": w, "questions": len(r["all@5"]),
                     **{m: round(float(np.mean(v)), 3) for m, v in r.items()}})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "importance_retrieval_comparison.csv", index=False)
    print(out.to_string(index=False))
    print("\nw = 0.43 is the current system's blend (0.7 similarity + 0.3 score).")


if __name__ == "__main__":
    main()