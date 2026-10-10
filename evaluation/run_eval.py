"""Full LoCoMo evaluation: adaptive memory vs truncation vs single summary.

For each chosen conversation:
  1. adaptive : every turn is fed through the real memory pipeline (categorize,
                score, contradiction check, store). Each question then retrieves
                the top-k memories and the model answers from only those.
  2. truncation-N : the model answers from only the last N turns.
  3. summary  : the model answers from one LLM-written summary of the whole
                conversation.

Per question we record: token F1 against the gold answer, an LLM-judged
correct/incorrect, the approximate context size (tokens), and for adaptive and
truncation whether the evidence turns were available (evidence recall).

GROQ'S FREE TIER IS THE BOTTLENECK. Roughly 30 requests/minute, 1,000
requests/day, 8,000 tokens/minute and 200,000 tokens/day per organization (check
your own limits in the Groq console). One full conversation needs far more than
a day's budget, so:
  * every Groq call (including the memory engine's own) goes through a gate
    that paces requests and waits out 429s using the wait time Groq reports;
  * when a DAILY limit is hit the run stops cleanly. Everything is saved,
    so run the same command again tomorrow and it resumes;
  * --max-turns evaluates a prefix of the conversation (first 100 turns, then
    200, ...). Ingestion is incremental, so nothing is redone, and the
    checkpoints give you the "accuracy vs conversation length" chart.

Run checkpoints in increasing order (100, then 200, ...).

Examples
  python evaluation/run_eval.py --dry-run --convs conv-26 --max-turns 100
  python evaluation/run_eval.py --convs conv-26 --max-turns 100 --max-questions 20
  python evaluation/run_eval.py --convs conv-26 --max-turns 200 --max-questions 20
  python evaluation/run_eval.py --convs conv-26 --fresh      (start over)
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation import locomo_utils as lu  # noqa: E402
from memory_engine import contradiction, pipeline, retrieve  # noqa: E402
from memory_engine.categorize import _get_client  # noqa: E402
from memory_engine.config import GROQ_MODEL  # noqa: E402
from memory_engine.store import MemoryStore  # noqa: E402

SLEEP = 0.0  # extra seconds between calls; the gate already paces requests

ANSWER_PROMPT = """You are answering a question about a long conversation \
between two people. You only have the memory notes below, not the full \
conversation. Dates in square brackets are when each message was sent; use them \
to work out relative times such as "yesterday" or "last week". If the notes do \
not contain the answer, reply "unknown". Answer in as few words as possible.

Memory notes:
{context}

Question: {question}
Answer:"""

JUDGE_PROMPT = """Question: {question}
Gold answer: {gold}
Model answer: {prediction}

Is the model answer correct? Count it as correct if it contains the key \
information of the gold answer. Be lenient about wording and about different \
formats of the same date, but an answer of "unknown" is never correct unless \
the gold answer is also unknown.
Respond with ONLY a JSON object: {{"correct": true}} or {{"correct": false}}"""

SUMMARY_PROMPT = """Summarize this part of a conversation between two people. \
Keep important facts, preferences, goals, decisions, events, names and dates. \
Do not add anything that is not in the conversation.

Conversation:
{conversation}"""

MERGE_PROMPT = """Below are summaries of consecutive parts of one long \
conversation. Merge them into a single summary. Keep important facts, \
preferences, goals, decisions, events, names and dates. Do not add anything \
that is not in the summaries.

{summaries}"""


# --------------------------------------------------------------------------
# Groq gate: pacing, 429 handling, daily-limit stop, usage counting
# --------------------------------------------------------------------------
class BudgetExhausted(BaseException):
    """Groq's daily limit (or a very long wait) was hit.

    Derives from BaseException on purpose: the memory engine wraps its LLM
    calls in `except Exception` and quietly falls back, which would let a run
    carry on and record garbage. This one cannot be swallowed that way."""


USAGE = {"requests": 0, "tokens": 0}

_RETRY_AFTER = re.compile(r"try again in\s+(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?", re.IGNORECASE)


def parse_retry_after(message: str):
    """Seconds Groq says to wait ('try again in 7m12.5s'), or None."""
    match = _RETRY_AFTER.search(message)
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds = match.groups()
    return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds or 0)


def is_daily_limit(message: str) -> bool:
    return bool(re.search(r"\((TPD|RPD)\)|per day", message, re.IGNORECASE))


def is_rate_limit(error: Exception) -> bool:
    return getattr(error, "status_code", None) == 429 or type(error).__name__ == "RateLimitError"


class GroqGate:
    """Wraps Groq's `create` call: keeps under the requests/minute and
    tokens/minute limits, waits out 429s, stops on the daily limit."""

    def __init__(self, rpm: float = 25, tpm: int = 7000):
        self.min_interval = 60.0 / rpm
        self.tpm = tpm
        self.last_call = 0.0
        self.window = []  # (timestamp, tokens) of calls in the last minute

    def _wait_for_capacity(self):
        now = time.time()
        wait = self.last_call + self.min_interval - now
        self.window = [(t, n) for t, n in self.window if now - t < 60]
        if sum(n for _, n in self.window) >= self.tpm:
            wait = max(wait, self.window[0][0] + 60 - now)
        if wait > 0:
            time.sleep(wait)

    def call(self, original, *args, **kwargs):
        for attempt in range(8):
            self._wait_for_capacity()
            self.last_call = time.time()
            try:
                response = original(*args, **kwargs)
            except Exception as e:
                if not is_rate_limit(e):
                    raise
                message = str(e)
                wait = parse_retry_after(message)
                if is_daily_limit(message) or (wait is not None and wait > 300):
                    raise BudgetExhausted(message[:500]) from e
                wait = wait + 1 if wait is not None else min(5 * 2 ** attempt, 60)
                print(f"    rate limited: {message[:200]}\n    waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            tokens = getattr(getattr(response, "usage", None), "total_tokens", 0) or 0
            USAGE["requests"] += 1
            USAGE["tokens"] += tokens
            self.window.append((time.time(), tokens))
            return response
        raise BudgetExhausted("Groq's rate limit did not clear after 8 attempts")


def install_gate(rpm: float, tpm: int) -> None:
    """Routes every Groq call (this script's and the memory engine's) through one gate."""
    gate = GroqGate(rpm, tpm)
    for get_client in (_get_client, contradiction._get_client):
        client = get_client()
        original = client.chat.completions.create
        client.chat.completions.create = lambda *a, _original=original, **k: gate.call(_original, *a, **k)


# --------------------------------------------------------------------------
# LLM access for this script. Swapped for a fake in --dry-run.
# --------------------------------------------------------------------------
def groq_chat(prompt: str, max_tokens: int, retries: int = 4) -> str:
    client = _get_client()
    delay = 4
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                return text
            raise RuntimeError("empty response (reasoning used all the tokens?)")
        except Exception as e:  # network error, empty answer ... (429s are handled by the gate)
            message = str(e)
            if "request too large" in message.lower():
                raise RuntimeError(
                    f"Request too large for Groq's per-minute limit: {message[:300]}\n"
                    f"Use a smaller --summary-chunk."
                ) from e
            if attempt == retries - 1:
                raise RuntimeError(f"Groq kept failing: {message[:300]}. Run again to resume.") from e
            print(f"    groq error ({type(e).__name__}: {message[:200]}); retry in {delay}s")
            time.sleep(delay)
            delay = min(delay * 2, 60)


_llm = groq_chat


def ask(prompt: str, max_tokens: int = 1024) -> str:
    text = _llm(prompt, max_tokens)
    if SLEEP:
        time.sleep(SLEEP)
    return text


# --------------------------------------------------------------------------
# small jsonl helpers
# --------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# adaptive memory: ingest a conversation (or a prefix) through the real pipeline
# --------------------------------------------------------------------------
def process_with_retry(memory_id: str, text: str, retries: int = 4) -> dict:
    delay = 4
    for attempt in range(retries):
        result = pipeline.process_message(memory_id, text)
        if result["action"] != "error":
            return result
        if attempt == retries - 1:
            raise RuntimeError(f"Categorizer kept failing: {result['reason'][:300]}. Run again to resume.")
        print(f"    categorizer error ({result['reason'][:200]}); retry in {delay}s")
        time.sleep(delay)
        delay = min(delay * 2, 60)


def ingest(conv_id: str, turns: list[dict], out_dir: Path, fresh: bool) -> list[dict]:
    memory_id = f"eval-{conv_id}"
    log_path = out_dir / f"{conv_id}__ingest.jsonl"
    store = pipeline.get_store()

    if fresh:
        store.delete_conversation(memory_id)
        log_path.unlink(missing_ok=True)

    log = read_jsonl(log_path)
    expected = sum(1 for r in log if r["action"] == "added")
    actual = len(store.get_all(memory_id))
    if expected != actual:
        raise RuntimeError(
            f"{conv_id}: the ingest log says {expected} stored memories but the database "
            f"has {actual}. Run again with --fresh."
        )

    done = {r["dia_id"] for r in log}
    beyond = done - {t["dia_id"] for t in turns}
    if beyond:
        raise RuntimeError(
            f"{conv_id}: memory already contains {len(beyond)} turns beyond this checkpoint, "
            f"so questions would see the future. Run checkpoints in increasing order, "
            f"or start over with --fresh."
        )

    todo = [t for t in turns if t["dia_id"] not in done]
    print(f"  ingesting {conv_id}: {len(done)} turns already done, {len(todo)} to go")

    for n, turn in enumerate(todo, 1):
        result = process_with_retry(memory_id, lu.format_turn(turn))
        append_jsonl(log_path, {
            "dia_id": turn["dia_id"],
            "action": result["action"],
            "category": result.get("category"),
            "reasoning": result.get("reasoning") or result.get("reason"),
            "llm_error": result.get("llm_error"),
        })
        if n % 25 == 0:
            print(f"    {n}/{len(todo)} turns   (~{USAGE['requests']} requests, ~{USAGE['tokens']} tokens so far)")
        if SLEEP:
            time.sleep(SLEEP)

    log = read_jsonl(log_path)
    errors = sum(1 for r in log if r.get("llm_error"))
    if errors:
        print(f"  WARNING: {errors} turns were stored without a working contradiction check "
              f"(the judge call failed). Consider --fresh.")
    return log


# --------------------------------------------------------------------------
# answering + scoring
# --------------------------------------------------------------------------
def judge_answer(question: str, gold: str, prediction: str):
    raw = ask(JUDGE_PROMPT.format(question=question, gold=gold, prediction=prediction), 1024)
    match = re.search(r'"correct"\s*:\s*(true|false)', raw, re.IGNORECASE)
    return match.group(1).lower() == "true" if match else None


def answer_and_score(q: dict, context: str, use_judge: bool):
    prediction = ask(ANSWER_PROMPT.format(context=context, question=q["question"]), 2048)
    f1 = lu.token_f1(prediction, q["answer"])
    correct = judge_answer(q["question"], q["answer"], prediction) if use_judge else None
    return prediction, f1, correct


def run_method(conv_id, method, questions, context_fn, out_dir, use_judge):
    path = out_dir / f"{conv_id}__{method}.jsonl"
    done = {r["idx"] for r in read_jsonl(path)}
    todo = [q for q in questions if q["idx"] not in done]
    print(f"  {method}: {len(done)} questions already done, {len(todo)} to go")

    for n, q in enumerate(todo, 1):
        context, extras = context_fn(q)
        prediction, f1, correct = answer_and_score(q, context, use_judge)
        append_jsonl(path, {
            "conv": conv_id,
            "method": method,
            "idx": q["idx"],
            "category": q["category"],
            "category_name": lu.CATEGORY_NAMES[q["category"]],
            "question": q["question"],
            "gold": q["answer"],
            "prediction": prediction,
            "f1": round(f1, 4),
            "judge_correct": correct,
            "context_tokens": lu.approx_tokens(context),
            **extras,
        })
        if n % 10 == 0:
            print(f"    {n}/{len(todo)} questions   (~{USAGE['requests']} requests, ~{USAGE['tokens']} tokens so far)")


def pick_questions(questions: list[dict], limit, seed: int = 0) -> list[dict]:
    """Same fixed random subset for every method, so they are compared on
    exactly the same questions."""
    if not limit or limit >= len(questions):
        return questions
    chosen = random.Random(seed).sample(questions, limit)
    return sorted(chosen, key=lambda q: q["idx"])


def build_summary(turns: list[dict], chunk_size: int = 50) -> str:
    lines = [lu.format_turn(t) for t in turns]
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    partial = [ask(SUMMARY_PROMPT.format(conversation="\n".join(c)), 1024) for c in chunks]
    if len(partial) == 1:
        return partial[0]
    return ask(MERGE_PROMPT.format(summaries="\n\n".join(partial)), 1024)


# --------------------------------------------------------------------------
# dry run: fake LLM + fake embeddings, to test the whole flow offline
# --------------------------------------------------------------------------
def install_dry_run_fakes():
    import hashlib

    import numpy as np

    def fake_embed(text):
        vec = np.zeros(64)
        for word in text.lower().split():
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else vec / norm

    def fake_categorize(text):
        body = text.split("]", 1)[-1].split(":", 1)[-1]
        return {"category": "recent_chat" if len(body.split()) < 6 else "fact", "confidence": 0.9}

    def fake_relation(old, new):
        return {"relation": "consistent", "reasoning": "dry-run"}

    def fake_llm(prompt, max_tokens):
        if "Is the model answer correct" in prompt:
            return '{"correct": false}'
        return "dry-run answer"

    global _llm
    pipeline.categorize = fake_categorize
    pipeline.embed = fake_embed
    contradiction.embed = fake_embed
    contradiction.check_relation = fake_relation
    retrieve.embed = fake_embed
    _llm = fake_llm


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def restrict_to_common(rows_by_method: dict) -> dict:
    """Keeps, for every method, only the questions that ALL methods answered,
    so methods are always compared on exactly the same questions."""
    common = None
    for rows in rows_by_method.values():
        keys = {(r["conv"], r["idx"]) for r in rows}
        common = keys if common is None else common & keys
    common = common or set()
    return {m: [r for r in rows if (r["conv"], r["idx"]) in common]
            for m, rows in rows_by_method.items()}


def quick_summary(qa_dir: Path, convs: list[str]) -> None:
    print("\n=== Quick summary (see summarize_results.py for charts and tables) ===")
    methods = sorted({p.stem.split("__", 1)[1] for c in convs for p in qa_dir.glob(f"{c}__*.jsonl")})
    rows_by_method = {m: [r for c in convs for r in read_jsonl(qa_dir / f"{c}__{m}.jsonl")] for m in methods}
    rows_by_method = {m: rows for m, rows in rows_by_method.items() if rows}
    compared = restrict_to_common(rows_by_method)
    print("(each method is scored only on the questions that every method answered)")
    print(f"{'method':<16}{'n':>5}{'F1':>8}{'judge':>8}{'ctx tokens':>12}{'ev@5':>8}")
    for method, rows in compared.items():
        if not rows:
            print(f"{method:<16}{0:>5}   (no question answered by all methods yet)")
            continue
        n = len(rows)
        f1 = sum(r["f1"] for r in rows) / n
        judged = [r["judge_correct"] for r in rows if r["judge_correct"] is not None]
        judge = f"{sum(judged) / len(judged):.2f}" if judged else "-"
        tokens = sum(r["context_tokens"] for r in rows) / n
        ev = [r["ev_all_at5"] for r in rows if "ev_all_at5" in r]
        ev_s = f"{sum(ev) / len(ev):.2f}" if ev else "-"
        print(f"{method:<16}{n:>5}{f1:>8.3f}{judge:>8}{tokens:>12.0f}{ev_s:>8}")
def clear_conversation_results(base_dir: Path, conv_id: str) -> None:
    pipeline.get_store().delete_conversation(f"eval-{conv_id}")
    folders = [base_dir, base_dir / "full", *base_dir.glob("t[0-9]*")]
    for folder in folders:
        if folder.is_dir():
            for p in folder.glob(f"{conv_id}__*"):
                p.unlink()


def evaluate_conversation(conv_id, entry, args, base_dir, qa_dir):
    turns = lu.conversation_turns(entry, include_captions=not args.no_captions)
    if args.max_turns:
        turns = turns[:args.max_turns]
    by_id = {t["dia_id"]: t for t in turns}

    questions = lu.answerable_questions(entry)
    if args.max_turns:
        questions = lu.questions_within(questions, turns)
    available = len(questions)
    questions = pick_questions(questions, args.max_questions)
    use_judge = not args.no_llm_judge
    print(f"  {len(turns)} turns, {len(questions)} questions"
          f"{f' (of {available} answerable within these turns)' if args.max_turns else ''}")

    if args.fresh:
        # Clear the stored memories too, otherwise the ingest consistency
        # check sees "empty log, full database" and refuses to continue.
        clear_conversation_results(base_dir, conv_id)

    if "adaptive" in args.methods:
        ingest(conv_id, turns, base_dir, fresh=False)
        items = pipeline.get_store().get_all(f"eval-{conv_id}")
        stored = {i["text"] for i in items}

        def adaptive_context(q):
            ranked = retrieve.retrieve_relevant(q["question"], items, top_k=10)
            texts = [r["text"] for r in ranked]
            ev = lu.evidence_texts(by_id, q["evidence"])
            context = "\n".join(f"- {t}" for t in texts[:args.top_k]) or "(no relevant memory found)"
            return context, {
                "n_retrieved": len(texts),
                "ev_retained": bool(ev) and all(t in stored for t in ev),
                "ev_all_at3": lu.evidence_hit(ev, texts[:3], True),
                "ev_all_at5": lu.evidence_hit(ev, texts[:5], True),
                "ev_all_at10": lu.evidence_hit(ev, texts[:10], True),
                "ev_any_at10": lu.evidence_hit(ev, texts[:10], False),
            }

        run_method(conv_id, "adaptive", questions, adaptive_context, qa_dir, use_judge)

    if "truncation" in args.methods:
        for n_turns in args.trunc_turns:
            window_texts = [lu.format_turn(t) for t in turns[-n_turns:]]
            context = "\n".join(window_texts)

            def trunc_context(q, context=context, window_texts=window_texts):
                ev = lu.evidence_texts(by_id, q["evidence"])
                return context, {"ev_all_in_window": lu.evidence_hit(ev, window_texts, True)}

            run_method(conv_id, f"truncation-{n_turns}", questions, trunc_context, qa_dir, use_judge)

    if "summary" in args.methods:
        summary_path = qa_dir / f"{conv_id}__summary.txt"
        if summary_path.exists():
            summary = summary_path.read_text(encoding="utf-8")
        else:
            print("  building the conversation summary ...")
            summary = build_summary(turns, chunk_size=args.summary_chunk)
            summary_path.write_text(summary, encoding="utf-8")

        run_method(conv_id, "summary", questions, lambda q: (summary, {}), qa_dir, use_judge)


def main():
    global SLEEP
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(ROOT / "data" / "locomo10.json"))
    parser.add_argument("--convs", nargs="+", default=["conv-26"], help="sample ids, or 'all'")
    parser.add_argument("--methods", nargs="+", default=["adaptive", "truncation", "summary"],
                        choices=["adaptive", "truncation", "summary"])
    parser.add_argument("--max-turns", type=int, default=None,
                        help="evaluate only the first N turns (a checkpoint); run in increasing order")
    parser.add_argument("--trunc-turns", nargs="+", type=int, default=[10, 40],
                        help="window sizes for the truncation baseline")
    parser.add_argument("--top-k", type=int, default=5, help="memories given to the model (adaptive)")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="random subset per conversation (same for all methods)")
    parser.add_argument("--summary-chunk", type=int, default=50, help="turns per summary chunk")
    parser.add_argument("--no-llm-judge", action="store_true", help="only token F1, about half the Groq calls")
    parser.add_argument("--no-captions", action="store_true", help="ignore photo captions")
    parser.add_argument("--rpm", type=float, default=25, help="max Groq requests per minute (limit is 30)")
    parser.add_argument("--tpm", type=int, default=7000, help="max Groq tokens per minute (limit is 8000)")
    parser.add_argument("--sleep", type=float, default=0.0, help="extra seconds between calls")
    parser.add_argument("--out", default=str(ROOT / "evaluation" / "results"))
    parser.add_argument("--fresh", action="store_true", help="discard saved results for these conversations")
    parser.add_argument("--dry-run", action="store_true", help="fake LLM/embeddings, flow test only")
    args = parser.parse_args()

    SLEEP = args.sleep
    base_dir = Path(args.out) / ("dryrun" if args.dry_run else "")
    qa_dir = base_dir / (f"t{args.max_turns}" if args.max_turns else "full")
    qa_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        install_dry_run_fakes()
        print("DRY RUN: fake LLM and embeddings, the numbers below are meaningless.\n")
    else:
        install_gate(args.rpm, args.tpm)

    # Separate database so evaluation never touches the dev memory.
    pipeline._store = MemoryStore(db_path=str(base_dir / "eval_memory.db"))

    data = lu.load_locomo(args.data)
    by_sample = {e["sample_id"]: e for e in data}
    convs = list(by_sample) if args.convs == ["all"] else args.convs
    unknown = [c for c in convs if c not in by_sample]
    if unknown:
        sys.exit(f"Unknown conversation id(s): {unknown}. Available: {list(by_sample)}")

    stopped = False
    try:
        for conv_id in convs:
            print(f"\n=== {conv_id} ===")
            evaluate_conversation(conv_id, by_sample[conv_id], args, base_dir, qa_dir)
    except BudgetExhausted as e:
        stopped = True
        print(f"\nSTOPPED: Groq's limit was hit and will not clear soon.\n  {e}\n"
              f"Everything so far is saved. Run the same command again later (daily limits reset "
              f"every day) or use another Groq account's key, and it resumes where it stopped.")

    print(f"\nGroq usage this run: ~{USAGE['requests']} requests, ~{USAGE['tokens']} tokens "
          f"(free daily limits: 1,000 requests / 200,000 tokens per organization)")
    quick_summary(qa_dir, convs)
    if stopped:
        sys.exit(2)


if __name__ == "__main__":
    main()