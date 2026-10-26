"""Learned importance score (optional).

A small gradient boosting model trained on LoCoMo predicts how likely a message is
to be needed later. It uses only features of the text itself, because a live chat
has no session numbers or turn positions. The model file is created by
evaluation/train_text_importance.py. If the file is missing, everything here
returns None and the system falls back to the hand set scorer.
"""
import re
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "models" / "importance_text.joblib"

TEXT_FEATURES = ["words", "chars", "has_image", "has_digit", "cap_words",
                 "has_question", "first_person"]

_FIRST_PERSON = re.compile(r"^\s*(i|i'm|i've|i'll|my|we|we're|our)\b", re.I)
_CAPS = re.compile(r"\b[A-Z][a-z]+")
_DIGIT = re.compile(r"\d")

_model = None
_loaded = False


def text_features(text: str) -> list[float]:
    """Same definitions as evaluation/build_importance_data.py."""
    return [
        float(len(text.split())),
        float(len(text)),
        float("(shared a photo:" in text),
        float(bool(_DIGIT.search(text))),
        float(len(_CAPS.findall(text))),
        float("?" in text),
        float(bool(_FIRST_PERSON.match(text))),
    ]


def _load():
    global _model, _loaded
    if _loaded:
        return _model
    _loaded = True
    if MODEL_PATH.exists():
        import joblib
        _model = joblib.load(MODEL_PATH)["model"]
    return _model


def is_available() -> bool:
    return _load() is not None


def learned_importance(text: str):
    """Probability in [0, 1] that this message will be needed later, or None
    if no trained model file is present."""
    model = _load()
    if model is None:
        return None
    return float(model.predict_proba([text_features(text)])[0][1])