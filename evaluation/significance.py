"""How sure are we that adaptive memory is better? Paired comparison, no Groq calls.

Every method answers the SAME questions, so we compare per question:
  * judged accuracy: mean difference, 95 percent bootstrap interval, and an exact
    McNemar test (a sign test on the questions where only one method was right)
  * token F1: mean difference and 95 percent bootstrap interval
Also a cluster check: the same difference per conversation.

    python evaluation/significance.py
    python evaluation/significance.py --dir evaluation/results/t100 --base adaptive
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(0)


def load(folder):
    rows = []
    for p in sorted(Path(folder).glob("*__*.jsonl")):
        if p.stem.endswith("__ingest"):
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def boot_ci(d, n=10000):
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "evaluation" / "results" / "t100"))
    ap.add_argument("--base", default="adaptive")
    args = ap.parse_args()

    df = load(args.dir)
    df = df[df.judge_correct.notna()].copy()
    df["judge_correct"] = df.judge_correct.astype(float)
    piv_j = df.pivot_table(index=["conv", "idx"], columns="method", values="judge_correct")
    piv_f = df.pivot_table(index=["conv", "idx"], columns="method", values="f1")
    common = piv_j.dropna().index
    piv_j, piv_f = piv_j.loc[common], piv_f.loc[common]
    print(f"{len(common)} questions answered by every method, from {common.get_level_values(0).nunique()} conversations\n")
    print(f"{'comparison':<32}{'accuracy diff':>14}{'95% CI':>20}{'only base right':>17}{'only other right':>17}{'p (McNemar)':>13}")
    for m in [c for c in piv_j.columns if c != args.base]:
        d = (piv_j[args.base] - piv_j[m]).to_numpy()
        lo, hi = boot_ci(d)
        b = int(((piv_j[args.base] == 1) & (piv_j[m] == 0)).sum())
        c = int(((piv_j[args.base] == 0) & (piv_j[m] == 1)).sum())
        p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        print(f"{args.base + ' vs ' + m:<32}{d.mean():>+14.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>20}{b:>17}{c:>17}{p:>13.4f}")
    print(f"\n{'comparison':<32}{'token F1 diff':>14}{'95% CI':>20}")
    for m in [c for c in piv_f.columns if c != args.base]:
        d = (piv_f[args.base] - piv_f[m]).to_numpy()
        lo, hi = boot_ci(d)
        print(f"{args.base + ' vs ' + m:<32}{d.mean():>+14.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>20}")
    print("\nPer conversation, judged accuracy:")
    print(piv_j.groupby(level=0).mean().round(2).to_string())
    print("\nA small test: an interval that includes 0 means the difference could be chance.")


if __name__ == "__main__":
    main()