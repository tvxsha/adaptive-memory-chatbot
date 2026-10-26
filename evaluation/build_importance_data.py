"""T3 Step A: build one row per LoCoMo turn with simple features and a label.

Label  needed = 1 if the turn is evidence for at least one answerable question.
Features use only the turn itself and where it sits (no API calls).

Run from the project folder:
    python evaluation/build_importance_data.py
Output: evaluation/results/importance_table.csv
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from locomo_utils import (load_locomo, conversation_turns,  # noqa: E402
                          answerable_questions)

FIRST_PERSON = re.compile(r"^\s*(i|i'm|i've|i'll|my|we|we're|our)\b", re.I)
CAPS = re.compile(r"\b[A-Z][a-z]+")
DIGIT = re.compile(r"\d")


def build(data_path: str) -> pd.DataFrame:
    rows = []
    for entry in load_locomo(data_path):
        conv_id = entry["sample_id"]
        turns = conversation_turns(entry)
        needed = set()
        for q in answerable_questions(entry):
            needed.update(q["evidence"])
        speakers = []
        for t in turns:
            if t["speaker"] not in speakers:
                speakers.append(t["speaker"])
        sessions = sorted({t["session"] for t in turns},
                          key=lambda k: int(k.split("_")[1]))
        sess_len = {s: sum(1 for t in turns if t["session"] == s) for s in sessions}
        sess_pos = {s: 0 for s in sessions}
        n = len(turns)
        for i, t in enumerate(turns):
            text = t["text"]
            sess_pos[t["session"]] += 1
            words = len(text.split())
            rows.append({
                "conv": conv_id,
                "dia_id": t["dia_id"],
                "text": text,
                "needed": int(t["dia_id"] in needed),
                "words": words,
                "chars": len(text),
                "has_image": int("(shared a photo:" in text),
                "pos_in_session": sess_pos[t["session"]] / sess_len[t["session"]],
                "session_no": sessions.index(t["session"]) + 1,
                "pos_in_conv": i / max(n - 1, 1),
                "speaker_first": int(t["speaker"] == speakers[0]),
                "has_digit": int(bool(DIGIT.search(text))),
                "cap_words": len(CAPS.findall(text)),
                "has_question": int("?" in text),
                "first_person": int(bool(FIRST_PERSON.match(text))),
                "specificity": min(words / 30.0, 0.2),   # current hand set bonus
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    data = ROOT / "data" / "locomo10.json"
    df = build(str(data))
    out = ROOT / "evaluation" / "results"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "importance_table.csv", index=False)
    print(f"{len(df)} turns from {df.conv.nunique()} conversations")
    print(f"share needed (base rate): {df.needed.mean():.3f}")
    print(df.groupby("conv").needed.agg(["count", "sum", "mean"]).round(3))