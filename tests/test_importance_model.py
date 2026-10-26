"""Offline tests for the optional learned importance score."""
import numpy as np

from memory_engine import config, importance_model, retrieve


def test_text_features_values():
    f = dict(zip(importance_model.TEXT_FEATURES,
                 importance_model.text_features("I moved to Mumbai in 2023. Why?")))
    assert f["words"] == 7
    assert f["has_digit"] == 1.0
    assert f["has_question"] == 1.0
    assert f["first_person"] == 1.0
    assert f["cap_words"] == 2          # "I" is not matched, Mumbai and Why are
    assert f["has_image"] == 0.0


def test_missing_model_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(importance_model, "MODEL_PATH", tmp_path / "none.joblib")
    monkeypatch.setattr(importance_model, "_loaded", False)
    monkeypatch.setattr(importance_model, "_model", None)
    assert importance_model.learned_importance("hello") is None
    assert importance_model.is_available() is False


def _items():
    return [
        {"id": "a", "text": "short", "embedding": np.array([1.0, 0.0]), "score": 0.5},
        {"id": "b", "text": "long and detailed", "embedding": np.array([0.99, 0.14]), "score": 0.5},
    ]


def test_retrieval_unchanged_when_flag_off(monkeypatch):
    monkeypatch.setattr(retrieve, "embed", lambda q: np.array([1.0, 0.0]))
    monkeypatch.setattr(config, "USE_LEARNED_IMPORTANCE", False)
    assert [r["id"] for r in retrieve.retrieve_relevant("q", _items())] == ["a", "b"]


def test_learned_score_can_reorder_when_flag_on(monkeypatch):
    monkeypatch.setattr(retrieve, "embed", lambda q: np.array([1.0, 0.0]))
    monkeypatch.setattr(config, "USE_LEARNED_IMPORTANCE", True)
    monkeypatch.setattr(config, "LEARNED_IMPORTANCE_WEIGHT", 0.5)
    monkeypatch.setattr(importance_model, "is_available", lambda: True)
    monkeypatch.setattr(importance_model, "learned_importance",
                        lambda t: 0.9 if t.startswith("long") else 0.0)
    assert [r["id"] for r in retrieve.retrieve_relevant("q", _items())] == ["b", "a"]


def test_flag_on_without_model_falls_back(monkeypatch):
    monkeypatch.setattr(retrieve, "embed", lambda q: np.array([1.0, 0.0]))
    monkeypatch.setattr(config, "USE_LEARNED_IMPORTANCE", True)
    monkeypatch.setattr(importance_model, "is_available", lambda: False)
    assert [r["id"] for r in retrieve.retrieve_relevant("q", _items())] == ["a", "b"]