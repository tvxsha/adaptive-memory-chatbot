# Results for the slides

All numbers come from `evaluation/summarize_results.py` (charts in `docs/figures/`).
Setup: LoCoMo conversation conv-26, first 100 turns, 20 questions (adversarial category excluded).
Every method is scored only on the questions that all methods answered. The answering model and the
judge are both `openai/gpt-oss-120b` on Groq. Embeddings are computed locally (MiniLM via fastembed).

## 1. Main result: accuracy vs. context size

| Method | Judge accuracy | Token F1 | Context tokens (avg) | Accuracy per 1k tokens |
|---|---|---|---|---|
| **Adaptive memory (top-5)** | **0.50** | **0.459** | **268** | **1.86** |
| Truncation (last 40 turns) | 0.45 | 0.421 | 1,848 | 0.24 |
| Single summary | 0.30 | 0.178 | 345 | 0.87 |
| Truncation (last 10 turns) | 0.10 | 0.067 | 375 | 0.27 |

**Headline:** Adaptive memory matches or slightly beats keeping the last 40 turns while sending about
7x less context (268 vs 1,848 tokens). With a context budget of the same size (about 350 tokens), it clearly beats
both the summary and 10-turn truncation (0.50 vs 0.30 vs 0.10).

Suggested charts: `4_accuracy_vs_cost.png` (main slide), `1_accuracy.png`, `3_context_tokens.png`.

## 2. By question type

| Type | Adaptive | Summary | Trunc-10 | Trunc-40 |
|---|---|---|---|---|
| Single-hop (n=11) | **0.64** | 0.36 | 0.00 | 0.45 |
| Temporal (n=6) | 0.33 | 0.00 | 0.17 | **0.50** |
| Multi-hop (n=2) | 0.00 | 0.50 | 0.00 | 0.00 |
| Open-domain (n=1) | 1.00 | 1.00 | 1.00 | 1.00 |

Adaptive memory is strongest on single-hop recall of facts. It is weaker on temporal questions
(it stores individual statements, not a timeline) and on multi-hop questions (the answer needs
several memories at once). Chart: `5_by_category.png`. Multi-hop and open-domain have very few questions, so don't draw conclusions from them.

## 3. Contradiction handling

Test: 60 hand-written memory pairs (15 updates, 10 contradictions, 20 consistent, 15 unrelated).
The pipeline = cosine similarity gate (>= 0.35) then an LLM judge on the 3 most similar memories.

- Correct replacements: 23. **Wrong replacements: 0.** Correct keeps: 35. Missed updates: 2.
- Precision **1.00**, recall **0.92**.
- Both misses were caught by the similarity gate (similarity 0.345 and 0.259), not by the judge. Lowering the gate to
  0.25 would catch both, at the cost of one extra judge call.
- Caveat: the pairs were written by the author. Independent labelling by a teammate is still pending (see `docs/`).
  Data: `docs/figures/contradiction_eval.csv`.

## 4. Retrieval analysis

- Only 32 of 100 turns are stored as memories. 65 turns are filtered out as small talk ("recent chat").
- For questions that need more than one memory, the right memories appear in the top-5 results 39% of the time
  (41% if the filtered chit-chat is also stored) and in the top-10 about 66% of the time.
- Ablation: storing the filtered chit-chat did not help accuracy (judge 0.55 vs 0.50, F1 0.451 vs 0.459, at n=20
  that is a 1-question difference, i.e. noise). So the filter saves storage and cost without hurting accuracy.
- Ranking strategies (embeddings, BM25, hybrid) were within noise of each other. Chart: `6_retrieval_recall.png`.
  Data: `docs/figures/retrieval_analysis.csv`.

## 5. Limitations (say these out loud, they strengthen the talk)

- One conversation, first 100 turns, 20 questions. With n=20 the 95% interval on an accuracy of 0.50 is roughly
  +/-0.22, so adaptive vs truncation-40 is **a tie on accuracy**. The solid claim is the ~7x lower context cost.
- The same model answers and judges, so the judge can be biased toward its own answers. Token F1 is reported as an
  independent check, and it points the same way.
- The extension only supports ChatGPT. Claude and Gemini selectors are placeholders.
- Temporal and multi-hop questions are the clear weak spots and the obvious future work (store dates, retrieve several memories).

## 6. One-line takeaways for slides

1. About 7x less context than truncation-40, with equal accuracy.
2. Beats a single summary and short truncation at a similar token budget.
3. Contradiction handling: 0 wrong replacements, 92% of updates caught.
4. Filtering small talk saves cost with no accuracy loss.
5. Weak on temporal and multi-hop questions, which is future work.