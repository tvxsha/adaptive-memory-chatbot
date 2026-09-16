"""Basic tests for the parts of the memory engine that don't require a live
Groq API call — categorize.py and contradiction.py's LLM-calling functions
are exercised separately/manually since they need a real API key.
"""
import numpy as np

from memory_engine.score import score_importance, specificity_bonus
from memory_engine.store import MemoryStore
from memory_engine.retrieve import retrieve_relevant


def test_score_importance_ranks_fact_above_recent_chat():
    fact_score = score_importance("I live in Chennai and work as an engineer", "fact")
    chat_score = score_importance("lol okay", "recent_chat")
    assert fact_score > chat_score


def test_specificity_bonus_is_capped():
    long_text = " ".join(["word"] * 100)
    assert specificity_bonus(long_text) <= 0.2


def test_store_add_and_get_all(tmp_path):
    db_path = tmp_path / "test_memory.db"
    store = MemoryStore(db_path=str(db_path))
    embedding = np.random.rand(384)
    store.add("conv1", "I like tea", "preference", 0.8, embedding)

    items = store.get_all("conv1")
    assert len(items) == 1
    assert items[0]["text"] == "I like tea"
    assert items[0]["category"] == "preference"


def test_store_replace_overwrites_not_duplicates(tmp_path):
    db_path = tmp_path / "test_memory.db"
    store = MemoryStore(db_path=str(db_path))
    embedding = np.random.rand(384)
    item_id = store.add("conv1", "I live in Delhi", "fact", 0.9, embedding)
    store.replace(item_id, "I live in Mumbai", "fact", 0.9, embedding)

    items = store.get_all("conv1")
    assert len(items) == 1
    assert items[0]["text"] == "I live in Mumbai"


def test_export_snapshot_excludes_embeddings(tmp_path):
    db_path = tmp_path / "test_memory.db"
    store = MemoryStore(db_path=str(db_path))
    store.add("conv1", "I like tea", "preference", 0.8, np.random.rand(384))

    snapshot = store.export_snapshot("conv1")
    assert "memory" in snapshot
    assert "embedding" not in snapshot["memory"][0]


def test_retrieve_relevant_returns_empty_for_no_items():
    result = retrieve_relevant("what do I like", [], top_k=5)
    assert result == []
