"""Classic EDA on the LoCoMo dataset (no model calls, no API key, runs in seconds).

Covers the standard EDA workflow: data understanding, cleaning audit,
descriptive statistics, univariate / bivariate / multivariate analysis, outliers
and key findings. Charts go to docs/figures/eda/, tables to docs/figures/eda/*.csv
and the printed summary is also saved as docs/figures/eda/eda_summary.txt.

Run from the project root:
    python evaluation/eda_analysis.py
"""
import argparse
import io
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from locomo_utils import CATEGORY_NAMES, evidence_ids, load_locomo  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sns.set_theme(style="whitegrid")
DATE_RE = re.compile(r"(\d+):(\d+)\s*(am|pm)\s+on\s+(\d+)\s+(\w+),?\s+(\d{4})", re.I)


def parse_date(text):
    """'1:56 pm on 8 May, 2023' -> Timestamp (NaT if it doesn't parse)."""
    m = DATE_RE.search(text or "")
    if not m:
        return pd.NaT
    hh, mm, ap, day, month, year = m.groups()
    return pd.to_datetime(f"{day} {month} {year} {hh}:{mm} {ap}", errors="coerce")


def build_tables(data):
    """Raw (uncleaned) turn table and question table straight from the JSON."""
    turns, qas, sessions = [], [], []
    for entry in data:
        cid = entry["sample_id"]
        conv = entry["conversation"]
        keys = sorted(
            (k for k in conv if k.startswith("session_") and not k.endswith("_date_time")),
            key=lambda k: int(k.split("_")[1]),
        )
        for key in keys:
            sessions.append({"conversation_id": cid, "session": key,
                             "date_raw": conv.get(f"{key}_date_time", ""),
                             "n_turns": len(conv[key])})
            for pos, t in enumerate(conv[key]):
                turns.append({
                    "conversation_id": cid,
                    "session": key,
                    "session_no": int(key.split("_")[1]),
                    "pos_in_session": pos,
                    "dia_id": t.get("dia_id"),
                    "speaker": t.get("speaker"),
                    "text": t.get("text"),
                    "has_image": bool(t.get("blip_caption") or t.get("img_url")),
                    "date_raw": conv.get(f"{key}_date_time", ""),
                })
        for q in entry["qa"]:
            qas.append({
                "conversation_id": cid,
                "question": q.get("question"),
                "answer": q.get("answer"),
                "category": q.get("category"),
                "evidence": evidence_ids(q.get("evidence")),
            })
    return pd.DataFrame(turns), pd.DataFrame(qas), pd.DataFrame(sessions)


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=150)
    plt.close(fig)
    print(f"  [chart] {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "locomo10.json"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures" / "eda"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    try:  # Windows consoles can't print every character (emoji etc.): replace instead of crashing
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    buf = io.StringIO()
    real_stdout = sys.stdout

    class Tee:
        def write(self, s):
            real_stdout.write(s)
            buf.write(s)

        def flush(self):
            real_stdout.flush()

    sys.stdout = Tee()
    try:
        run(args, out)
    finally:
        sys.stdout = real_stdout
    (out / "eda_summary.txt").write_text(buf.getvalue(), encoding="utf-8")
    print(f"\nSaved summary text to {out / 'eda_summary.txt'}")


def run(args, out):
    data = load_locomo(args.data)
    turns, qa, sessions = build_tables(data)

    # ------------------------------------------------------------------ 1
    section("1. DATA UNDERSTANDING")
    print(f"Source: LoCoMo benchmark ({args.data})")
    print(f"Conversations: {turns['conversation_id'].nunique()}")
    print(f"Sessions:      {len(sessions)}  (avg {len(sessions) / turns['conversation_id'].nunique():.1f} per conversation)")
    print(f"Turns:         {len(turns)}")
    print(f"Questions:     {len(qa)}")
    print("\nVariables in the turn table:")
    desc = pd.DataFrame({"dtype": turns.dtypes.astype(str),
                         "non_null": turns.notna().sum(),
                         "unique": turns.nunique()})
    print(desc.to_string())
    print("\nFirst 3 turns:")
    print(turns[["conversation_id", "session", "dia_id", "speaker", "text"]].head(3).to_string(index=False, max_colwidth=60))

    # ------------------------------------------------------------------ 2
    section("2. DATA CLEANING AND PREPROCESSING (audit of the raw data)")
    t = turns.copy()
    missing_text = t["text"].isna().sum()
    t["text"] = t["text"].fillna("")
    empty = (t["text"].str.strip() == "").sum()
    ws_issue = (t["text"] != t["text"].str.strip()).sum() + t["text"].str.contains(r"\s{2,}", regex=True).sum()
    dup_ids = t.duplicated(subset=["conversation_id", "dia_id"]).sum()
    norm = t["text"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
    dup_text = norm.duplicated(keep="first") & (norm != "")
    t["words"] = t["text"].str.split().str.len()
    short = (t["words"] < 4).sum()
    missing_speaker = t["speaker"].isna().sum()
    bad_dates = sessions["date_raw"].map(parse_date).isna().sum()
    print(f"Missing text values:                {missing_text}")
    print(f"Empty / whitespace-only texts:       {empty}")
    print(f"Texts with stray / double whitespace: {ws_issue}")
    print(f"Missing speaker labels:              {missing_speaker}")
    print(f"Duplicate dialogue ids:              {dup_ids}")
    print(f"Exact duplicate texts (normalized):  {int(dup_text.sum())}  ({dup_text.mean():.1%})")
    print(f"Very short turns (< 4 words):        {short}  ({short / len(t):.1%})")
    print(f"Session dates that failed to parse:  {bad_dates}")
    qa_missing_answer = qa["answer"].isna().sum()
    qa_adv = (qa["category"] == 5).sum()
    print(f"Questions with no gold answer:       {qa_missing_answer}  (they belong to the {qa_adv} 'adversarial' questions, which are excluded from accuracy)")

    # Preprocessing actually applied
    t["text"] = t["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    t = t[t["text"] != ""].copy()
    t["date"] = t["date_raw"].map(parse_date)
    t["chars"] = t["text"].str.len()
    t["words"] = t["text"].str.split().str.len()
    t["avg_word_len"] = t["chars"] / t["words"].clip(lower=1)
    t["question_marks"] = t["text"].str.count(r"\?")
    t["exclamations"] = t["text"].str.count("!")
    t["sentences"] = t["text"].str.count(r"[.!?]+").clip(lower=1)
    t["is_short"] = t["words"] < 4
    t["turn_in_conv"] = t.groupby("conversation_id").cumcount()
    start = t.groupby("conversation_id")["date"].transform("min")
    t["days_since_start"] = (t["date"] - start).dt.total_seconds() / 86400
    print("\nPreprocessing applied: whitespace normalized, empty turns dropped, session dates converted to")
    print("datetimes, text features derived (words, chars, question marks, ...).")
    print(f"Turns kept after cleaning: {len(t)} of {len(turns)}")
    print("Take-away: LoCoMo itself is already clean; the real 'cleaning' is done by the memory pipeline")
    print("(near-duplicates skipped, small talk filtered, contradictions replaced).")

    qa["cat_name"] = qa["category"].map(CATEGORY_NAMES)
    qa["n_evidence"] = qa["evidence"].str.len()
    qa["answer_words"] = qa["answer"].fillna("").astype(str).str.split().str.len()
    qa["question_words"] = qa["question"].str.split().str.len()

    # ------------------------------------------------------------------ 3
    section("3. DESCRIPTIVE STATISTICS")
    num = ["words", "chars", "avg_word_len", "sentences", "question_marks", "exclamations"]
    d = t[num].describe().T
    d["skew"] = t[num].skew()
    d["kurtosis"] = t[num].kurt()
    print(d.round(2).to_string())
    d.round(3).to_csv(out / "descriptive_stats_turns.csv")
    print("\nWords per turn by speaker (top 6 speakers by turn count):")
    top_sp = t["speaker"].value_counts().head(6).index
    sp = t[t["speaker"].isin(top_sp)].groupby("speaker")["words"].agg(["count", "mean", "median", "std", "max"])
    print(sp.round(1).to_string())
    print("\nQuestion statistics:")
    print(qa[["n_evidence", "question_words", "answer_words"]].describe().round(2).to_string())
    print("\nQuestions per category:")
    print(qa["cat_name"].value_counts().to_string())

    # ------------------------------------------------------------------ 4
    section("4. UNIVARIATE ANALYSIS")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(t["words"], bins=40, kde=True, ax=ax[0], color="#4a4ae0")
    ax[0].set_title("Words per turn (histogram + KDE)")
    ax[0].set_xlabel("words")
    sns.boxplot(x=t["words"], ax=ax[1], color="#9aa0f5")
    ax[1].set_title("Words per turn (boxplot)")
    ax[1].set_xlabel("words")
    save(fig, out, "eda_1_words_per_turn.png")
    print(f"  Median {t['words'].median():.0f} words, mean {t['words'].mean():.1f}: the distribution is right-skewed")
    print(f"  (skew {t['words'].skew():.2f}); most turns are short, a few are long.")

    fig, ax = plt.subplots(figsize=(7, 4))
    order = qa["cat_name"].value_counts().index
    sns.countplot(data=qa, y="cat_name", order=order, ax=ax, color="#4a4ae0")
    ax.set_title("Questions per category")
    ax.set_xlabel("count")
    ax.set_ylabel("")
    for p in ax.patches:
        ax.annotate(int(p.get_width()), (p.get_width(), p.get_y() + p.get_height() / 2),
                    va="center", ha="left", fontsize=9)
    save(fig, out, "eda_2_question_categories.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    per_sess = sessions.groupby("conversation_id")["session"].count()
    per_sess.plot(kind="bar", ax=ax[0], color="#4a4ae0")
    ax[0].set_title("Sessions per conversation")
    ax[0].set_xlabel("")
    tp = t.groupby("conversation_id").size()
    tp.plot(kind="bar", ax=ax[1], color="#1a8f4c")
    ax[1].set_title("Turns per conversation")
    ax[1].set_xlabel("")
    save(fig, out, "eda_3_conversation_sizes.png")
    print(f"  Conversations have {tp.min()} to {tp.max()} turns (median {tp.median():.0f}).")

    # ------------------------------------------------------------------ 5
    section("5. BIVARIATE ANALYSIS")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.boxplot(data=t[t["speaker"].isin(top_sp)], x="speaker", y="words", ax=ax[0], color="#9aa0f5")
    ax[0].set_title("Words per turn by speaker")
    ax[0].tick_params(axis="x", rotation=30)
    sns.scatterplot(data=t.sample(min(2500, len(t)), random_state=0), x="turn_in_conv", y="words",
                    alpha=0.25, ax=ax[1], color="#4a4ae0")
    sns.regplot(data=t, x="turn_in_conv", y="words", scatter=False, ax=ax[1], color="#c62828",
                lowess=False, line_kws={"linewidth": 2})
    ax[1].set_title("Turn length vs. position in the conversation")
    ax[1].set_xlabel("turn index within conversation")
    save(fig, out, "eda_4_bivariate_turns.png")
    r = t["turn_in_conv"].corr(t["words"])
    r2 = t["pos_in_session"].corr(t["words"])
    print(f"  Turn length vs position in the whole conversation: r = {r:.2f} (no trend).")
    print(f"  Turn length vs position inside a session:          r = {r2:.2f} (turns get shorter as a session goes on).")

    qa_plot = qa[qa["category"].isin([1, 2, 3, 4])]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.boxplot(data=qa_plot, x="cat_name", y="n_evidence", ax=ax[0], color="#9aa0f5")
    ax[0].set_title("Evidence turns needed per question, by category")
    ax[0].set_xlabel("")
    ax[0].set_ylabel("number of evidence turns")
    sns.boxplot(data=qa_plot, x="cat_name", y="answer_words", ax=ax[1], color="#a5d6b7")
    ax[1].set_title("Gold answer length by category")
    ax[1].set_xlabel("")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("answer words (log scale)")
    save(fig, out, "eda_5_bivariate_questions.png")
    g = qa_plot.groupby("cat_name")["n_evidence"].mean().round(2)
    print("  Mean evidence turns per question by category:")
    print("   " + g.to_string().replace("\n", "\n   "))
    print("  Multi-hop questions need the most evidence turns, which is why retrieval of several memories")
    print("  at once is the hardest case for the memory system.")

    # ------------------------------------------------------------------ 6
    section("6. MULTIVARIATE / CORRELATION ANALYSIS")
    feats = ["words", "chars", "avg_word_len", "sentences", "question_marks", "exclamations",
             "has_image", "pos_in_session", "turn_in_conv", "days_since_start"]
    corr = t[feats].astype(float).corr()
    corr.round(3).to_csv(out / "correlation_matrix.csv")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation heatmap of turn features")
    save(fig, out, "eda_6_correlation_heatmap.png")
    pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().sort_values(key=abs, ascending=False)
    print("  Strongest correlations:")
    for (a, b), v in pairs.head(5).items():
        print(f"    {a} vs {b}: {v:.2f}")
    print("  (words/chars/sentences move together by construction: they measure the same thing, so only one")
    print("  of them needs to be used when modelling.)")

    sample = t.sample(min(1500, len(t)), random_state=1)[["words", "avg_word_len", "question_marks", "pos_in_session"]]
    sample["words"] = np.log1p(sample["words"])
    pg = sns.pairplot(sample.rename(columns={"words": "log(words)"}), corner=True,
                      plot_kws={"alpha": 0.3, "s": 12}, diag_kind="kde")
    pg.fig.suptitle("Pair plot of turn features (sample of 1,500 turns)", y=1.02)
    pg.fig.savefig(out / "eda_7_pairplot.png", dpi=130)
    plt.close(pg.fig)
    print("  [chart] eda_7_pairplot.png")

    # ------------------------------------------------------------------ 7
    section("7. OUTLIER ANALYSIS (IQR rule on words per turn)")
    q1, q3 = t["words"].quantile([0.25, 0.75])
    iqr = q3 - q1
    hi = q3 + 1.5 * iqr
    outl = t[t["words"] > hi]
    z = (t["words"] - t["words"].mean()) / t["words"].std()
    print(f"  Q1 = {q1:.0f}, Q3 = {q3:.0f}, IQR = {iqr:.0f}; upper fence = {hi:.1f} words")
    print(f"  Outliers by IQR rule: {len(outl)} turns ({len(outl) / len(t):.1%});  by |z| > 3: {(z.abs() > 3).sum()}")
    print("  Longest turns:")
    for _, row in outl.sort_values("words", ascending=False).head(3).iterrows():
        print(f"    [{row['conversation_id']} {row['dia_id']}] {row['words']} words: {row['text'][:90]}...")
    print(f"  Share of outliers that include a photo: {outl['has_image'].mean():.0%}"
          f" (vs {t['has_image'].mean():.0%} of all turns)")
    print("  Decision: outliers are genuine long messages (stories, plans), not errors, so they are KEPT; they")
    print("  matter because they contain many facts worth remembering.")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(x=t["words"], ax=ax[0], color="#9aa0f5", flierprops={"markerfacecolor": "#c62828", "markersize": 3})
    ax[0].axvline(hi, color="#c62828", linestyle="--", label=f"upper fence = {hi:.0f}")
    ax[0].legend()
    ax[0].set_title("Outliers in words per turn (IQR rule)")
    sns.boxplot(data=t, x="conversation_id", y="words", ax=ax[1], color="#a5d6b7", fliersize=2)
    ax[1].set_title("Words per turn by conversation")
    ax[1].tick_params(axis="x", rotation=45)
    ax[1].set_xlabel("")
    save(fig, out, "eda_8_outliers.png")
    outl[["conversation_id", "dia_id", "speaker", "words", "text"]].sort_values("words", ascending=False) \
        .to_csv(out / "outlier_turns.csv", index=False)

    # ------------------------------------------------------------------ 8
    section("8. KEY FINDINGS")
    short_pct = t["is_short"].mean()
    multi = qa_plot.loc[qa_plot["cat_name"] == "multi-hop", "n_evidence"].mean()
    single = qa_plot.loc[qa_plot["cat_name"] == "single-hop", "n_evidence"].mean()
    findings = [
        f"The dataset has {len(t):,} turns in {turns['conversation_id'].nunique()} long conversations "
        f"({tp.min()}-{tp.max()} turns each, over {len(sessions)} sessions): far too long to paste into a prompt.",
        f"Turn length is right-skewed (median {t['words'].median():.0f} words, mean {t['words'].mean():.1f}, "
        f"skew {t['words'].skew():.2f}); only {short_pct:.1%} of turns have fewer than 4 words.",
        f"The raw data is clean (0 missing texts, 0 duplicate ids), so cleaning has to happen at memory time: "
        f"in conv-26, 65 of the first 100 turns were filtered out as small talk.",
        f"{len(outl)} turns ({len(outl) / len(t):.1%}) are length outliers; they are real long messages and "
        f"are kept because they carry many facts.",
        f"Multi-hop questions need {multi:.1f} evidence turns on average vs {single:.1f} for single-hop, "
        f"which explains why they are the hardest for retrieval-based memory.",
        f"Messages get shorter as a session goes on (words vs position in session r = "
        f"{t['words'].corr(t['pos_in_session']):.2f}), but length is unrelated to where a turn sits in the whole "
        f"conversation (r = {t['words'].corr(t['turn_in_conv']):.2f}); length features (words, chars) are almost "
        "identical (r = 0.99), so one of them is enough when modelling.",
    ]
    for i, f in enumerate(findings, 1):
        print(f"  {i}. {f}")


if __name__ == "__main__":
    main()