"""Edge-case tests for the full pipeline.

These run completely offline: the Groq calls (categorize, check_relation) and
the Hugging Face embedding call are replaced with deterministic fakes, so no
API keys or internet are needed. What's being tested is the pipeline's own
logic: guards, skipping, replacing, duplicate handling.
"""
import hashlib

import numpy as np
import pytest

from memory_engine import contradiction, pipeline
from memory_engine.store import MemoryStore


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
def fake_embed(text: str) -> np.ndarray:
    """Deterministic bag-of-words embedding. Texts that share words get a high
    cosine similarity; identical texts get exactly 1.0.
    """
    vec = np.zeros(64)
    for word in text.lower().split():
        idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % 64
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


@pytest.fixture
def engine(tmp_path, monkeypatch):
    """A pipeline wired to a temp database and fake LLM/embedding calls.

    Returns a small control object so each test can choose what the fake
    categorizer and fake contradiction judge say.
    """
    store = MemoryStore(db_path=str(tmp_path / "test.db"))
    monkeypatch.setattr(pipeline, "_store", store)

    state = {
        "category": "fact",
        "relation": "consistent",
        "categorize_inputs": [],
        "relation_calls": 0,
    }

    def fake_categorize(text):
        state["categorize_inputs"].append(text)
        return {"category": state["category"], "confidence": 0.9}

    def fake_check_relation(old_text, new_text):
        state["relation_calls"] += 1
        return {"relation": state["relation"], "reasoning": "fake judge"}

    monkeypatch.setattr(pipeline, "categorize", fake_categorize)
    monkeypatch.setattr(pipeline, "embed", fake_embed)
    monkeypatch.setattr(contradiction, "embed", fake_embed)
    monkeypatch.setattr(contradiction, "check_relation", fake_check_relation)

    state["store"] = store
    return state


# --------------------------------------------------------------------------
# Input guards
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad_input", ["", "   ", "\n\t  \n", None])
def test_empty_or_whitespace_message_is_skipped(engine, bad_input):
    result = pipeline.process_message("c1", bad_input)
    assert result["action"] == "skip"
    assert engine["categorize_inputs"] == []  # never reached the LLM
    assert engine["store"].get_all("c1") == []


def test_very_long_message_is_truncated_before_llm(engine):
    long_text = "I live in Delhi. " * 1000  # ~17,000 chars
    pipeline.process_message("c1", long_text)
    sent = engine["categorize_inputs"][0]
    assert len(sent) <= pipeline.MAX_MESSAGE_CHARS


def test_surrounding_whitespace_is_stripped(engine):
    pipeline.process_message("c1", "   I live in Delhi   \n")
    assert engine["store"].get_all("c1")[0]["text"] == "I live in Delhi"


# --------------------------------------------------------------------------
# Categories
# --------------------------------------------------------------------------
def test_recent_chat_is_not_stored(engine):
    engine["category"] = "recent_chat"
    result = pipeline.process_message("c1", "haha okay cool")
    assert result["action"] == "skip"
    assert engine["store"].get_all("c1") == []


def test_ambiguous_message_with_low_confidence_still_returns_valid_result(engine):
    engine["category"] = "fact"
    result = pipeline.process_message("c1", "maybe")
    assert result["action"] in {"added", "skip"}


# --------------------------------------------------------------------------
# Duplicates, contradictions, updates
# --------------------------------------------------------------------------
def test_same_message_twice_is_stored_once(engine):
    first = pipeline.process_message("c1", "My favorite food is biryani")
    second = pipeline.process_message("c1", "My favorite food is biryani")
    assert first["action"] == "added"
    assert second["action"] == "skip"
    assert len(engine["store"].get_all("c1")) == 1
    # Exact duplicates shouldn't cost an LLM judgment call
    assert engine["relation_calls"] == 0


def test_contradiction_replaces_instead_of_duplicating(engine):
    pipeline.process_message("c1", "I live in Delhi")
    engine["relation"] = "contradicts"
    result = pipeline.process_message("c1", "I live in Mumbai")

    assert result["action"] == "replaced"
    items = engine["store"].get_all("c1")
    assert len(items) == 1
    assert items[0]["text"] == "I live in Mumbai"


def test_update_relation_also_replaces(engine):
    pipeline.process_message("c1", "I live in Delhi")
    engine["relation"] = "updates"
    pipeline.process_message("c1", "I live in Mumbai")
    assert [i["text"] for i in engine["store"].get_all("c1")] == ["I live in Mumbai"]


def test_consistent_statements_are_both_kept(engine):
    pipeline.process_message("c1", "I live in Delhi")
    engine["relation"] = "consistent"
    result = pipeline.process_message("c1", "I live in Delhi near the metro station")
    assert result["action"] == "added"
    assert len(engine["store"].get_all("c1")) == 2


def test_unrelated_message_skips_the_llm_judge(engine):
    pipeline.process_message("c1", "I live in Delhi")
    result = pipeline.process_message("c1", "zebra quantum banana orbit")
    assert result["action"] == "added"
    assert engine["relation_calls"] == 0  # similarity too low to bother the LLM
    assert len(engine["store"].get_all("c1")) == 2


def test_conversations_do_not_leak_into_each_other(engine):
    pipeline.process_message("c1", "I live in Delhi")
    engine["relation"] = "contradicts"
    result = pipeline.process_message("c2", "I live in Mumbai")
    assert result["action"] == "added"  # nothing in c2 to contradict
    assert len(engine["store"].get_all("c1")) == 1
    assert len(engine["store"].get_all("c2")) == 1


# --------------------------------------------------------------------------
# LLM failure handling (categorize / check_relation fallbacks)
# --------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]


class _FakeClient:
    def __init__(self, content=None, raises=False):
        self._content, self._raises = content, raises
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        if self._raises:
            raise RuntimeError("simulated API failure")
        return _FakeResponse(self._content)


def test_categorize_falls_back_on_garbage_output(monkeypatch):
    from memory_engine import categorize as cat
    monkeypatch.setattr(cat, "_get_client", lambda: _FakeClient("not json at all"))
    assert cat.categorize("hello")["category"] == "recent_chat"


def test_categorize_falls_back_on_api_error(monkeypatch):
    from memory_engine import categorize as cat
    monkeypatch.setattr(cat, "_get_client", lambda: _FakeClient(raises=True))
    result = cat.categorize("hello")
    assert result["category"] == "recent_chat"
    assert result["confidence"] == 0.0
    assert "simulated API failure" in result["error"]


def test_categorize_rejects_unknown_category(monkeypatch):
    from memory_engine import categorize as cat
    monkeypatch.setattr(
        cat, "_get_client",
        lambda: _FakeClient('{"category": "banana", "confidence": 0.9}'),
    )
    assert cat.categorize("hello")["category"] == "recent_chat"


def test_categorize_handles_markdown_fenced_json(monkeypatch):
    from memory_engine import categorize as cat
    fenced = '```json\n{"category": "goal", "confidence": 0.8}\n```'
    monkeypatch.setattr(cat, "_get_client", lambda: _FakeClient(fenced))
    assert cat.categorize("I want to learn Spanish")["category"] == "goal"


def test_check_relation_falls_back_to_unrelated_on_garbage(monkeypatch):
    monkeypatch.setattr(contradiction, "_get_client", lambda: _FakeClient("???"))
    assert contradiction.check_relation("a", "b")["relation"] == "unrelated"


# --------------------------------------------------------------------------
# Judging more than the single best candidate
# --------------------------------------------------------------------------
def test_resolve_judges_second_best_candidate_too(monkeypatch):
    """The memory a statement updates is not always the most similar one.
    Judging only the top match would miss this update (found in the smoke test:
    "I just moved to Mumbai" matched a Goa flight better than the Delhi fact).
    """
    monkeypatch.setattr(contradiction, "embed", lambda t: np.array([1.0, 0.0]))
    items = [
        {"id": "decoy", "text": "I booked a flight to Goa", "embedding": np.array([0.9, 0.436])},
        {"id": "target", "text": "I live in Delhi", "embedding": np.array([0.6, 0.8])},
    ]
    judged = []

    def judge(old, new):
        judged.append(old)
        if "Delhi" in old:
            return {"relation": "updates", "reasoning": "moved cities"}
        return {"relation": "consistent", "reasoning": "different topic"}

    monkeypatch.setattr(contradiction, "check_relation", judge)
    result = contradiction.resolve("I just moved to Mumbai", items)

    assert result["action"] == "replace"
    assert result["target_id"] == "target"
    assert judged == ["I booked a flight to Goa", "I live in Delhi"]


def test_resolve_stops_after_max_candidates(monkeypatch):
    monkeypatch.setattr(contradiction, "embed", lambda t: np.array([1.0, 0.0]))
    items = [
        {"id": f"m{i}", "text": f"memory {i}", "embedding": np.array([0.9 - i * 0.05, 0.4])}
        for i in range(6)
    ]
    judged = []

    def judge(old, new):
        judged.append(old)
        return {"relation": "consistent", "reasoning": "fine"}

    monkeypatch.setattr(contradiction, "check_relation", judge)
    result = contradiction.resolve("something new", items)

    assert result["action"] == "add"
    assert len(judged) == contradiction.MAX_CANDIDATES_TO_JUDGE
# --------------------------------------------------------------------------
# LLM failures must be visible, not silent
# --------------------------------------------------------------------------
def test_categorizer_failure_returns_error_and_stores_nothing(engine, monkeypatch):
    monkeypatch.setattr(
        pipeline, "categorize",
        lambda text: {"category": "recent_chat", "confidence": 0.0, "error": "boom"},
    )
    result = pipeline.process_message("c1", "I live in Delhi")
    assert result["action"] == "error"
    assert "boom" in result["reason"]
    assert engine["store"].get_all("c1") == []


def test_judge_failure_is_flagged_but_memory_is_still_stored(engine, monkeypatch):
    pipeline.process_message("c1", "I live in Delhi")
    monkeypatch.setattr(
        contradiction, "check_relation",
        lambda old, new: {"relation": "unrelated", "reasoning": "", "error": "judge down"},
    )
    result = pipeline.process_message("c1", "I live in Delhi near the metro station")
    assert result["action"] == "added"
    assert result["llm_error"] == "judge down"
    assert len(engine["store"].get_all("c1")) == 2


def test_check_relation_returns_error_field_on_api_failure(monkeypatch):
    monkeypatch.setattr(contradiction, "_get_client", lambda: _FakeClient(raises=True))
    result = contradiction.check_relation("a", "b")
    assert result["relation"] == "unrelated"
    assert "simulated API failure" in result["error"]


def test_recent_chat_is_skipped_by_default(tmp_path, monkeypatch):
    from memory_engine import pipeline, config
    from memory_engine.store import MemoryStore
    monkeypatch.setattr(pipeline, "_store", MemoryStore(db_path=str(tmp_path / "t.db")))
    monkeypatch.setattr(pipeline, "categorize", lambda t: {"category": "recent_chat", "confidence": 0.9})
    monkeypatch.setattr(config, "STORE_RECENT_CHAT", False)
    result = pipeline.process_message("c", "haha yeah totally")
    assert result["action"] == "skip"
    assert pipeline.get_store().get_all("c") == []


def test_recent_chat_is_stored_when_enabled(tmp_path, monkeypatch):
    import numpy as np
    from memory_engine import pipeline, config
    from memory_engine.store import MemoryStore
    monkeypatch.setattr(pipeline, "_store", MemoryStore(db_path=str(tmp_path / "t.db")))
    monkeypatch.setattr(pipeline, "categorize", lambda t: {"category": "recent_chat", "confidence": 0.9})
    monkeypatch.setattr(pipeline, "embed", lambda t: np.ones(4) / 2)
    monkeypatch.setattr(config, "STORE_RECENT_CHAT", True)
    result = pipeline.process_message("c", "haha yeah totally")
    assert result["action"] == "added"
    items = pipeline.get_store().get_all("c")
    assert len(items) == 1 and items[0]["category"] == "recent_chat"