"""Offline tests for retrieval relevance filtering."""
import numpy as np

from memory_engine import retrieve


def _items():
    return [
        {"id": "close", "text": "close", "embedding": np.array([1.0, 0.0]), "score": 0.5},
        {"id": "far", "text": "far", "embedding": np.array([0.0, 1.0]), "score": 1.5},
    ]


def test_irrelevant_item_is_dropped_even_with_high_importance(monkeypatch):
    monkeypatch.setattr(retrieve, "embed", lambda q: np.array([1.0, 0.0]))
    results = retrieve.retrieve_relevant("query", _items(), top_k=5)
    assert [r["id"] for r in results] == ["close"]


def test_returns_empty_when_nothing_is_relevant(monkeypatch):
    monkeypatch.setattr(retrieve, "embed", lambda q: np.array([0.0, 0.0, 1.0])[:2] * 0)
    assert retrieve.retrieve_relevant("query", _items()) == []


def test_min_similarity_can_be_overridden(monkeypatch):
    monkeypatch.setattr(retrieve, "embed", lambda q: np.array([1.0, 0.0]))
    results = retrieve.retrieve_relevant("query", _items(), min_similarity=-1.0)
    assert len(results) == 2


def test_top_k_still_applies(monkeypatch):
    monkeypatch.setattr(retrieve, "embed", lambda q: np.array([1.0, 0.0]))
    items = [
        {"id": f"m{i}", "text": "x", "embedding": np.array([1.0, 0.0]), "score": 1.0}
        for i in range(6)
    ]
    assert len(retrieve.retrieve_relevant("query", items, top_k=3)) == 3


def test_no_items_returns_empty():
    assert retrieve.retrieve_relevant("query", []) == []