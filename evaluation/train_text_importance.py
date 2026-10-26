"""Trains the importance model that the live system can use (text features only).

Trains on all 10 LoCoMo conversations and saves memory_engine/models/importance_text.joblib.
Honest evaluation of this feature set is done by importance_nested_weight.py --text-only,
which never trains on the conversation it scores.

    python evaluation/train_text_importance.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from memory_engine.importance_model import TEXT_FEATURES, MODEL_PATH  # noqa: E402
from train_importance import make_models  # noqa: E402


def main():
    df = pd.read_csv(ROOT / "evaluation" / "results" / "importance_table.csv")
    model = make_models()["Gradient Boosting"].fit(
        df[TEXT_FEATURES].to_numpy(dtype=float), df["needed"].to_numpy())
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": TEXT_FEATURES}, MODEL_PATH)
    print(f"trained on {len(df)} turns, saved {MODEL_PATH}")


if __name__ == "__main__":
    main()