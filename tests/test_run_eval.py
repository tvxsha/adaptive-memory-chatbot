"""Offline tests for the evaluation runner's own logic (retry, resume,
parsing, summary chunking). The LLM is always faked."""
import numpy as np
import pytest

from evaluation import run_eval
from memory_engine import pipeline
from memory_engine.store import MemoryStore


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    monkeypatch.setattr(run_eval.time, "sleep", lambda s: None)
    monkeypatch.setattr(run_eval, "SLEEP", 0.0)


# ---- question sampling ---------------------------------------------------
def test_pick_questions_is_deterministic_sorted_subset():
    qs = [{"idx": i} for i in range(50)]
    a = run_eval.pick_questions(qs, 10)
    b = run_eval.pick_questions(qs, 10)
    assert a == b
    assert len(a) == 10
    assert [q["idx"] for q in a] == sorted(q["idx"] for q in a)


def test_pick_questions_without_limit_returns_all():
    qs = [{"idx": i} for i in range(5)]
    assert run_eval.pick_questions(qs, None) == qs
    assert run_eval.pick_questions(qs, 99) == qs


# ---- judge parsing -------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ('{"correct": true}', True),
    ('{"correct": false}', False),
    ('Sure! {"correct": True}', True),
    ("I think it is fine", None),  # unparseable -> None, never a guess
])
def test_judge_answer_parsing(monkeypatch, raw, expected):
    monkeypatch.setattr(run_eval, "_llm", lambda prompt, max_tokens: raw)
    assert run_eval.judge_answer("q", "gold", "pred") is expected


# ---- groq_chat retry/backoff --------------------------------------------
class _Client:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        message = type("M", (), {"content": outcome})()
        return type("R", (), {"choices": [type("C", (), {"message": message})()]})()


def test_groq_chat_retries_then_succeeds(monkeypatch):
    client = _Client([RuntimeError("429 rate limit"), RuntimeError("429 rate limit"), "hello"])
    monkeypatch.setattr(run_eval, "_get_client", lambda: client)
    assert run_eval.groq_chat("p", 100) == "hello"
    assert client.calls == 3


def test_groq_chat_empty_response_counts_as_failure(monkeypatch):
    client = _Client(["", "real answer"])
    monkeypatch.setattr(run_eval, "_get_client", lambda: client)
    assert run_eval.groq_chat("p", 100) == "real answer"


def test_groq_chat_gives_up_with_clear_message(monkeypatch):
    client = _Client([RuntimeError("down")] * 3)
    monkeypatch.setattr(run_eval, "_get_client", lambda: client)
    with pytest.raises(RuntimeError, match="Run again to resume"):
        run_eval.groq_chat("p", 100, retries=3)
    assert client.calls == 3


# ---- categorizer retry ---------------------------------------------------
def test_process_with_retry_recovers(monkeypatch):
    results = [
        {"action": "error", "reason": "429"},
        {"action": "added", "category": "fact"},
    ]
    monkeypatch.setattr(pipeline, "process_message", lambda cid, text: results.pop(0))
    assert run_eval.process_with_retry("c", "text")["action"] == "added"


def test_process_with_retry_raises_when_it_keeps_failing(monkeypatch):
    monkeypatch.setattr(
        pipeline, "process_message", lambda cid, text: {"action": "error", "reason": "429"}
    )
    with pytest.raises(RuntimeError, match="Run again to resume"):
        run_eval.process_with_retry("c", "text", retries=3)


# ---- summary chunking ----------------------------------------------------
def _turns(n):
    return [{"dia_id": f"D1:{i}", "speaker": "A", "text": f"t{i}", "session": "s", "date": "d"}
            for i in range(n)]


def test_summary_chunks_then_merges(monkeypatch):
    prompts = []
    monkeypatch.setattr(run_eval, "_llm", lambda p, m: prompts.append(p) or "summary")
    run_eval.build_summary(_turns(5), chunk_size=2)  # 3 chunks + 1 merge
    assert len(prompts) == 4
    assert "consecutive parts" in prompts[-1]


def test_summary_single_chunk_skips_merge(monkeypatch):
    prompts = []
    monkeypatch.setattr(run_eval, "_llm", lambda p, m: prompts.append(p) or "summary")
    assert run_eval.build_summary(_turns(3), chunk_size=100) == "summary"
    assert len(prompts) == 1


# ---- ingest: resume and consistency check ------------------------------
@pytest.fixture
def fake_pipeline(tmp_path, monkeypatch):
    store = MemoryStore(db_path=str(tmp_path / "t.db"))
    monkeypatch.setattr(pipeline, "_store", store)
    processed = []

    def fake_process(memory_id, text):
        processed.append(text)
        store.add(memory_id, text, "fact", 1.0, np.array([1.0, 0.0]))
        return {"action": "added", "category": "fact", "reasoning": "ok"}

    monkeypatch.setattr(pipeline, "process_message", fake_process)
    return store, processed


def test_ingest_resumes_without_reprocessing(tmp_path, fake_pipeline):
    store, processed = fake_pipeline
    turns = _turns(4)
    run_eval.ingest("conv-x", turns[:2], tmp_path, fresh=False)
    assert len(processed) == 2
    run_eval.ingest("conv-x", turns, tmp_path, fresh=False)  # resume with all 4
    assert len(processed) == 4  # only the 2 new turns were processed
    assert len(store.get_all("eval-conv-x")) == 4


def test_ingest_detects_log_database_mismatch(tmp_path, fake_pipeline):
    store, processed = fake_pipeline
    run_eval.ingest("conv-x", _turns(3), tmp_path, fresh=False)
    store.delete_conversation("eval-conv-x")  # database lost, log kept
    with pytest.raises(RuntimeError, match="--fresh"):
        run_eval.ingest("conv-x", _turns(3), tmp_path, fresh=False)


def test_ingest_fresh_starts_over(tmp_path, fake_pipeline):
    store, processed = fake_pipeline
    run_eval.ingest("conv-x", _turns(3), tmp_path, fresh=False)
    run_eval.ingest("conv-x", _turns(3), tmp_path, fresh=True)
    assert len(processed) == 6
    assert len(store.get_all("eval-conv-x")) == 3


# ---- Groq gate: retry-after parsing, pacing, 429 handling ---------------
class _RateLimited(Exception):
    status_code = 429


@pytest.fixture
def gate_env(monkeypatch):
    sleeps = []
    monkeypatch.setattr(run_eval.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setitem(run_eval.USAGE, "requests", 0)
    monkeypatch.setitem(run_eval.USAGE, "tokens", 0)
    return sleeps


def _response(tokens):
    return type("R", (), {"usage": type("U", (), {"total_tokens": tokens})()})()


@pytest.mark.parametrize("message,expected", [
    ("Please try again in 7m12.5s.", 432.5),
    ("Please try again in 2.5s.", 2.5),
    ("Please try again in 1h2m3s.", 3723),
    ("Please try again in 800ms.", None),   # milliseconds: fall back to default backoff
    ("something else entirely", None),
])
def test_parse_retry_after(message, expected):
    assert run_eval.parse_retry_after(message) == expected


def test_daily_limit_detection():
    assert run_eval.is_daily_limit("... on tokens per day (TPD): Limit 200000 ...")
    assert run_eval.is_daily_limit("... on requests per day (RPD): Limit 1000 ...")
    assert not run_eval.is_daily_limit("... on tokens per minute (TPM): Limit 8000 ...")


def test_gate_waits_out_a_per_minute_limit_then_succeeds(gate_env):
    sleeps = gate_env
    outcomes = [_RateLimited("... tokens per minute (TPM) ... Please try again in 3s."), _response(120)]

    def original(*args, **kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    gate = run_eval.GroqGate(rpm=1000, tpm=7000)
    response = gate.call(original)
    assert response.usage.total_tokens == 120
    assert 4.0 in sleeps  # waited the 3s Groq asked for, plus 1s margin
    assert run_eval.USAGE == {"requests": 1, "tokens": 120}


def test_gate_stops_on_daily_limit_and_cannot_be_swallowed(gate_env):
    def original(*args, **kwargs):
        raise _RateLimited("... tokens per day (TPD): Limit 200000 ... Please try again in 7m12s.")

    gate = run_eval.GroqGate()
    with pytest.raises(run_eval.BudgetExhausted):
        gate.call(original)
    # The memory engine catches `Exception` and falls back silently. The stop
    # signal must get through that.
    assert not issubclass(run_eval.BudgetExhausted, Exception)


def test_gate_treats_a_very_long_wait_as_exhausted(gate_env):
    def original(*args, **kwargs):
        raise _RateLimited("rate limited. Please try again in 45m0s.")

    with pytest.raises(run_eval.BudgetExhausted):
        run_eval.GroqGate().call(original)


def test_gate_passes_other_errors_through(gate_env):
    def original(*args, **kwargs):
        raise ValueError("bad request")

    with pytest.raises(ValueError):
        run_eval.GroqGate().call(original)


def test_gate_waits_when_the_token_budget_for_the_minute_is_used(gate_env):
    sleeps = gate_env
    gate = run_eval.GroqGate(rpm=1000, tpm=100)
    gate.call(lambda: _response(100))      # uses the whole minute's budget
    gate.call(lambda: _response(10))       # must wait for the window to clear
    assert max(sleeps) > 50


def test_gate_paces_requests(gate_env):
    sleeps = gate_env
    gate = run_eval.GroqGate(rpm=30, tpm=10**9)  # one request every 2 seconds
    gate.call(lambda: _response(1))
    gate.call(lambda: _response(1))
    assert any(s > 1.5 for s in sleeps)


def test_budget_exhausted_passes_through_the_categorizer(monkeypatch):
    """categorize() swallows ordinary exceptions; the stop signal must not be swallowed."""
    from memory_engine import categorize as cat

    class Boom:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": self})()

        def create(self, **kwargs):
            raise run_eval.BudgetExhausted("daily limit")

    monkeypatch.setattr(cat, "_get_client", lambda: Boom())
    with pytest.raises(run_eval.BudgetExhausted):
        cat.categorize("hello")


def test_install_gate_routes_engine_calls_through_the_gate(monkeypatch, gate_env):
    from memory_engine import contradiction

    def make_client():
        client = type("C", (), {})()
        client.chat = type("Chat", (), {})()
        client.chat.completions = type("Comp", (), {})()
        client.chat.completions.create = lambda **kw: _response(50)
        return client

    clients = [make_client(), make_client()]
    monkeypatch.setattr(run_eval, "_get_client", lambda: clients[0])
    monkeypatch.setattr(contradiction, "_get_client", lambda: clients[1])
    run_eval.install_gate(rpm=1000, tpm=10**9)

    clients[0].chat.completions.create(model="m")
    clients[1].chat.completions.create(model="m")
    assert run_eval.USAGE == {"requests": 2, "tokens": 100}


# ---- checkpoints must run in increasing order -----------------------------
def test_ingest_refuses_to_go_back_to_a_smaller_checkpoint(tmp_path, fake_pipeline):
    store, processed = fake_pipeline
    run_eval.ingest("conv-x", _turns(4), tmp_path, fresh=False)
    with pytest.raises(RuntimeError, match="increasing order"):
        run_eval.ingest("conv-x", _turns(2), tmp_path, fresh=False)


# ---- methods are compared on the same questions only ----------------------
def test_restrict_to_common_keeps_only_questions_every_method_answered():
    def row(conv, idx):
        return {"conv": conv, "idx": idx}

    rows = {
        "adaptive": [row("c", 1), row("c", 2), row("c", 3)],
        "summary": [row("c", 2), row("c", 3), row("c", 4)],
        "truncation-10": [row("c", 3), row("c", 2)],
    }
    result = run_eval.restrict_to_common(rows)
    for method_rows in result.values():
        assert {r["idx"] for r in method_rows} == {2, 3}


def test_restrict_to_common_does_not_mix_conversations():
    rows = {
        "a": [{"conv": "c1", "idx": 1}, {"conv": "c2", "idx": 1}],
        "b": [{"conv": "c1", "idx": 1}],
    }
    result = run_eval.restrict_to_common(rows)
    assert result["a"] == [{"conv": "c1", "idx": 1}]