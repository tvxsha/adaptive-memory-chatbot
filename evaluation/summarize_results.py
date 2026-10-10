"""Turns the evaluation results (JSONL) into tables and charts.

Usage (from the project root):
    python evaluation/summarize_results.py
    python evaluation/summarize_results.py --dir evaluation/results/t100

Writes evaluation/figures/*.png and evaluation/figures/summary.csv.
Every method is scored only on the questions that ALL methods answered.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ORDER_HINT = ["adaptive", "summary", "truncation-10", "truncation-40"]
COLORS = {"adaptive": "#4a4ae0", "summary": "#c27803", "truncation-10": "#9aa0b4", "truncation-40": "#6b7088"}


def load_rows(folder: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(folder.glob("*__*.jsonl")):
        if p.stem.endswith("__summary.txt"):
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def restrict_to_common(df: pd.DataFrame) -> pd.DataFrame:
    keys = None
    for _, g in df.groupby("method"):
        k = set(zip(g["conv"], g["idx"]))
        keys = k if keys is None else keys & k
    keys = keys or set()
    mask = [(c, i) in keys for c, i in zip(df["conv"], df["idx"])]
    return df[mask].copy()


def method_order(methods):
    return sorted(methods, key=lambda m: ORDER_HINT.index(m) if m in ORDER_HINT else 99)


def bar_chart(df, value, title, ylabel, path, order, ylim=None):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    means = df.groupby("method")[value].mean().reindex(order)
    ax.bar(order, means.values, color=[COLORS.get(m, "#888") for m in order])
    for i, v in enumerate(means.values):
        ax.text(i, v, f"{v:.2f}" if v < 100 else f"{v:.0f}", ha="center", va="bottom", fontsize=10)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    sns.despine()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="folder with the *.jsonl files (default: newest evaluation/results/t*)")
    ap.add_argument("--out", default="evaluation/figures")
    args = ap.parse_args()

    if args.dir:
        folder = Path(args.dir)
    else:
        cands = sorted(Path("evaluation/results").glob("t[0-9]*"), key=lambda p: p.stat().st_mtime)
        if not cands:
            raise SystemExit("No results found under evaluation/results/t*")
        folder = cands[-1]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_rows(folder)
    if df.empty:
        raise SystemExit(f"No rows in {folder}")
    raw_counts = df.groupby("method").size().to_dict()
    df = restrict_to_common(df)
    order = method_order(df["method"].unique())
    n = df[df["method"] == order[0]].shape[0] if order else 0
    print(f"Folder: {folder}\nRows per method before filtering: {raw_counts}\nQuestions answered by every method: {n}\n")
    if n == 0:
        raise SystemExit("No question was answered by all methods yet.")

    df["judge_correct"] = pd.to_numeric(df["judge_correct"], errors="coerce")
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.7)

    table = df.groupby("method").agg(
        n=("idx", "count"),
        f1=("f1", "mean"),
        judge_accuracy=("judge_correct", "mean"),
        context_tokens=("context_tokens", "mean"),
    ).reindex(order)
    if "ev_all_at5" in df:
        table["evidence_recall_at5"] = df.groupby("method")["ev_all_at5"].mean()
    table["accuracy_per_1k_tokens"] = table["judge_accuracy"] / (table["context_tokens"] / 1000)
    table.round(3).to_csv(out / "summary.csv")
    print(table.round(3).to_string(), "\n")

    bar_chart(df, "judge_correct", f"Answer correctness (LLM-judged), n={n}", "fraction correct", out / "1_accuracy.png", order, (0, 1))
    bar_chart(df, "f1", f"Token F1 vs gold answer, n={n}", "F1", out / "2_f1.png", order, (0, 1))
    bar_chart(df, "context_tokens", "Context given to the model (approx. tokens)", "tokens", out / "3_context_tokens.png", order)

    # accuracy vs context size
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in order:
        ax.scatter(table.loc[m, "context_tokens"], table.loc[m, "judge_accuracy"], s=160, color=COLORS.get(m, "#888"), label=m, zorder=3)
        ax.annotate(m, (table.loc[m, "context_tokens"], table.loc[m, "judge_accuracy"]), textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xlabel("context tokens per question (lower is cheaper)")
    ax.set_ylabel("judge accuracy")
    ax.set_title("Accuracy vs. context cost")
    ax.set_ylim(0, 1.05)
    sns.despine()
    fig.tight_layout()
    fig.savefig(out / "4_accuracy_vs_cost.png", dpi=200)
    plt.close(fig)

    # per question type
    cat = df.pivot_table(index="category_name", columns="method", values="judge_correct", aggfunc="mean")[order]
    counts = df[df["method"] == order[0]].groupby("category_name").size()
    cat.index = [f"{c} (n={counts.get(c, 0)})" for c in cat.index]
    cat.round(3).to_csv(out / "by_category.csv")
    print(cat.round(2).to_string(), "\n")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cat.plot(kind="bar", ax=ax, color=[COLORS.get(m, "#888") for m in cat.columns], width=0.8)
    ax.set_ylabel("fraction correct")
    ax.set_ylim(0, 1.05)
    ax.set_title("Correctness by question type")
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    ax.legend(frameon=False, fontsize=8)
    sns.despine()
    fig.tight_layout()
    fig.savefig(out / "5_by_category.png", dpi=200)
    plt.close(fig)

    # retrieval quality (adaptive): did the needed memories reach the model?
    ad = df[df["method"] == "adaptive"]
    ev_cols = [c for c in ["ev_all_at3", "ev_all_at5", "ev_all_at10", "ev_any_at10"] if c in ad]
    if ev_cols and len(ad):
        vals = ad[ev_cols].mean()
        fig, ax = plt.subplots(figsize=(6, 3.8))
        labels = [c.replace("ev_all_at", "all evidence in top-").replace("ev_any_at", "any evidence in top-") for c in ev_cols]
        ax.bar(labels, vals.values, color=COLORS["adaptive"])
        for i, v in enumerate(vals.values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
        ax.set_ylim(0, 1.05)
        ax.set_title("Adaptive retrieval: evidence recall")
        plt.xticks(rotation=15, ha="right")
        sns.despine()
        fig.tight_layout()
        fig.savefig(out / "6_retrieval_recall.png", dpi=200)
        plt.close(fig)

    print(f"Saved charts and CSVs to {out}/")


if __name__ == "__main__":
    main()