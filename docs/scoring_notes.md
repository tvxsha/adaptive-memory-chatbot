# Scoring & retrieval tuning notes

Working notes for the tuning phase. The "Measured so far" section comes from
real runs (scripts in `scripts/`); everything under "Open questions" is still a
hypothesis. Numbers marked provisional should be re-tuned on Vrishti's
contradiction test set. Edit freely.

## Current values

`memory_engine/score.py`

| category       | weight |
|----------------|--------|
| fact           | 0.90   |
| goal           | 0.85   |
| decision       | 0.85   |
| preference     | 0.80   |
| completed_task | 0.50   |
| recent_chat    | 0.20 (never stored) |

Plus a specificity bonus of `min(words / 30, 0.2)`, so stored scores run from
about 0.7 to 1.1. These weights are placeholders, not measured.

`memory_engine/contradiction.py`
- `SIMILARITY_THRESHOLD = 0.35` (provisional): below this, a new statement is
  never compared against an old one.
- `DUPLICATE_THRESHOLD = 0.97`: above this, the statement is a repeat and skipped.
- `MAX_CANDIDATES_TO_JUDGE = 3`: how many similar memories go to the LLM judge.

`memory_engine/retrieve.py`
- `combined = 0.7 * similarity + 0.3 * score`
- `MIN_SIMILARITY = 0.15` (provisional): items below this are dropped, so
  retrieval can return fewer than `top_k`, or nothing.

Models: Groq `openai/gpt-oss-120b` for categorizing and judging; Hugging Face
`all-MiniLM-L6-v2` for embeddings.

## Measured so far

**Embedding similarity (scripts/check_similarity.py, 16 hand-picked pairs)**
- Real updates ("I live in Delhi" -> "I just moved to Mumbai"): 0.46-0.54.
- Same-topic statements that do NOT conflict ("I live in Delhi" -> "I live in
  Delhi near the metro station"): 0.36-0.45.
- Unrelated pairs: 0.14 or lower.
- No single threshold separates updates from same-topic non-conflicts. The LLM
  judge is the real safeguard, and the threshold is only a cost filter. That is
  why it was lowered from 0.55 to 0.35: at 0.55 real updates never reached the judge.

**LLM judge (scripts/check_judge.py, same 16 pairs)**
- 16/16 correct after tightening the prompt (first prompt: 14/16, it called
  elaborations like "I practice Spanish on Duolingo" an update and would have
  deleted a correct memory).
- Caveat: 16 easy, hand-picked pairs is not an accuracy estimate. The hard
  cases are what Vrishti's test set needs to cover.

**Judging only the best match was a bug (smoke test)**
- "Actually I just moved to Mumbai last week" was most similar to "I booked a
  flight to Goa", not "I live in Delhi", so the Delhi fact survived. Judging the
  top 3 candidates fixed it. The memory a statement updates is not always the
  most similar one.

**Retrieval cutoff (smoke test)**
- Without a cutoff, "where does the user live?" returned an unrelated laptop
  decision (similarity 0.087) just to fill `top_k`.
- With `MIN_SIMILARITY = 0.15` that is gone, but a correct memory ("I live in
  Mumbai and I work as a data analyst", 0.116) is also dropped, while irrelevant
  ones still get through for another query ("what is the user studying?"
  returns the laptop at 0.189 and a Mumbai fact at 0.205).
- Conclusion: MiniLM similarity between a short question and a statement is too
  noisy for one fixed number to separate everything.

**Speed (scripts/time_llm.py)**
- gpt-oss-120b: categorize ~0.7s, judge ~1.0s per call. gpt-oss-20b was only
  ~0.1s faster with the same accuracy, so we stay on 120b.
- A message costs 1 categorize call plus 0-3 judge calls, so roughly 1-4s per
  message before the embedding call. Groq free-tier rate limits matter more than
  latency for bulk evaluation.

**Failure handling**
- LLM errors used to be swallowed (everything looked like filler). They are now
  logged and returned as `action: "error"` (categorizer) or an `llm_error` flag
  (judge), so an outage can't masquerade as "the system forgot everything".

## Open questions

1. **Scores barely discriminate.** Stored scores sit between 0.7 and 1.1, so
   `0.3 * score` adds at most about 0.12 of spread to the combined value.
   Retrieval is effectively similarity-ranked. Check the spread on real
   conversations; widen the weights or drop the term if it stays narrow.

2. **Is a fixed retrieval cutoff good enough?** See the measurements above.
   Alternatives: a cutoff relative to the best match, or an LLM reranker over
   the top few results.

3. **completed_task at 0.5.** Is a finished task less useful later than a
   preference? Probably depends on the question type (LongMemEval has
   categories that would show this).

4. **Elaborations pile up.** "I live in Mumbai" and "I just moved to Mumbai"
   are both stored (the judge says consistent). Harmless but redundant; a merge
   step could be added later.

5. **gpt-oss is a reasoning model and not fully deterministic** even at
   temperature 0. Re-run the judge a few times on the test set to see how stable
   it is.

## Experiments to run (once Vrishti's contradiction test set exists)

- Sweep `SIMILARITY_THRESHOLD` over 0.25-0.55; plot precision and recall of
  contradiction detection per value.
- Compare `MAX_CANDIDATES_TO_JUDGE` of 1, 2 and 3 on the test set.
- Sweep `MIN_SIMILARITY` against the retrieval questions; compare with and
  without the score term, and with a reranker.
- Compare the heuristic scorer against an LLM-assigned importance score.
- Turn on `recency_decay()` and check whether it helps or hurts recall on long
  conversations.
- Re-run the judge several times to measure run-to-run variation.