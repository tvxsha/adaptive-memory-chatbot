"""Ranking experiments: NO Groq calls (embeddings run locally, free).

Question: which way of ranking memories finds the evidence turns best?
Every strategy ranks the SAME pool (all ingested turns), so this isolates
ranking quality from the question of what got stored.

    python evaluation/retrieval_experiments.py --conv conv-26 --max-turns 100

Metrics (higher is better), over questions whose evidence is in the checkpoint:
    all@k : ALL evidence turns are in the top-k   (strict)
    any@k : AT LEAST ONE evidence turn is in the top-k (lenient)
"""
import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

import locomo_utils as lu  # noqa: E402
from memory_engine.embeddings import embed  # noqa: E402

STOP = set("a an the and or of to in on at for with is are was were be been it that this i you he she they we my your his her "
           "their our me him them do did does done have has had what when where who whom how why which about as by from not no so "
           "if but just also would could can will".split())


def tokens(text):
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [Counter(tokens(d)) for d in docs]
        self.len = [sum(c.values()) for c in self.docs]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter(w for c in self.docs for w in c)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query):
        q = tokens(query)
        out = np.zeros(len(self.docs))
        for i, c in enumerate(self.docs):
            for w in q:
                if w in c:
                    f = c[w]
                    out[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


def rrf(*score_arrays, k=60):
    """Reciprocal-rank fusion of several score arrays."""
    total = np.zeros(len(score_arrays[0]))
    for s in score_arrays:
        order = np.argsort(-s)
        ranks = np.empty(len(s))
        ranks[order] = np.arange(1, len(s) + 1)
        total += 1.0 / (k + ranks)
    return total


def strip_date(turn):
    return f"{turn['speaker']}: {turn['text']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "locomo10.json"))
    ap.add_argument("--conv", default="conv-26")
    ap.add_argument("--max-turns", type=int, default=100)
    args = ap.parse_args()

    entry = {e["sample_id"]: e for e in lu.load_locomo(args.data)}[args.conv]
    turns = lu.conversation_turns(entry)[: args.max_turns]
    by_id = {t["dia_id"]: i for i, t in enumerate(turns)}
    questions = [q for q in lu.questions_within(lu.answerable_questions(entry), turns)
                 if any(e in by_id for e in q["evidence"])]

    full_texts = [lu.format_turn(t) for t in turns]
    short_texts = [strip_date(t) for t in turns]
    print(f"embedding {len(turns)} turns (twice) locally ...")
    emb_full = np.stack([embed(t) for t in full_texts])
    emb_short = np.stack([embed(t) for t in short_texts])
    bm25 = BM25(short_texts)

    strategies = {
        "embeddings (date prefix)": lambda q, qe: emb_full @ qe,
        "embeddings (no date)": lambda q, qe: emb_short @ qe,
        "keyword BM25": lambda q, qe: bm25.scores(q),
        "hybrid RRF (emb + BM25)": lambda q, qe: rrf(emb_short @ qe, bm25.scores(q)),
        "hybrid RRF (3 signals)": lambda q, qe: rrf(emb_full @ qe, emb_short @ qe, bm25.scores(q)),
    }
    results = {name: {m: [] for m in ("all@3", "all@5", "all@10", "any@5", "any@10")} for name in strategies}
    cats = {name: {} for name in strategies}

    for q in questions:
        ev = {by_id[e] for e in q["evidence"] if e in by_id}
        qe = embed(q["question"])
        for name, fn in strategies.items():
            order = list(np.argsort(-fn(q["question"], qe)))
            r = results[name]
            for k in (3, 5, 10):
                r[f"all@{k}"].append(ev <= set(order[:k]))
            for k in (5, 10):
                r[f"any@{k}"].append(bool(ev & set(order[:k])))
            cats[name].setdefault(lu.CATEGORY_NAMES[q["category"]], []).append(ev <= set(order[:5]))

    n = len(questions)
    print(f"\n{n} questions, pool of {len(turns)} turns (random guessing would score all@5 ~ {5 / len(turns):.2f} per evidence turn)\n")
    cols = ["all@3", "all@5", "all@10", "any@5", "any@10"]
    print(f"{'strategy':<28}" + "".join(f"{c:>8}" for c in cols))
    for name in strategies:
        print(f"{name:<28}" + "".join(f"{np.mean(results[name][c]):>8.2f}" for c in cols))
    print("\nall@5 by question type:")
    for name in strategies:
        print(f"  {name:<28}" + "  ".join(f"{c}={np.mean(v):.2f}(n={len(v)})" for c, v in sorted(cats[name].items())))


if __name__ == "__main__":
    main()