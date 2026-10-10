# Roadmap — Adaptive Memory for AI Chatbots (final stretch)

**Deadline: Monday 12 Oct.** Written Saturday 10 Oct night, checked against `main` (PRs #1 to #8 merged).
Owners: **Tvisha** (backend, extension, demo, final runs) · **Vrishti** (evaluation, EDA, contradiction set) · **Peehu** (slides and report text).

## Rules for everyone
- One branch per task, then a PR. No Claude attribution lines in commits or PRs.
- Groq's free quota is **per account (organization)**: about 200k tokens/day, shared across all keys of the same account.
  Everyone uses their **own** Groq key and keeps `GROQ_MODEL=openai/gpt-oss-120b` in `.env`, so results are comparable.
- Pull the latest `main` and run `pip install -r requirements.txt` before starting.
- **No evaluation runs on Monday.** Tvisha's Groq quota stays free for the live demo (every "Add to memory" click uses it).
- If a command prints `STOPPED: Groq's limit...`, nothing is lost: run the same command later and it resumes.
- Tell the group chat immediately when a task that blocks someone else is finished.

---

## Tvisha — backend, extension, demo, final runs

### Already done

**Project setup and process**
- Project scaffold, GitHub repo, one-branch-per-change and PR workflow (8 PRs merged), collaborators added
- Roadmap, evaluation plan and task split; reviewed Vrishti's EDA commit before merging

**Memory engine (PRs #1 to #6)**
- Pipeline hardening: step-by-step logging, input guards (empty and over-long messages), duplicate skip, offline tests, smoke test script
- Moved the LLM to `openai/gpt-oss-120b` after `llama-3.3-70b-versatile` was retired; ran a speed test against gpt-oss-20b and kept 120b
- Contradiction detection tuned with measured data: similarity gate lowered from 0.55 to 0.35 (real updates scored 0.39 to 0.54),
  judge prompt tightened to separate "update" from "elaboration", token limit raised for the reasoning model
- Wrote `check_similarity.py` and `check_judge.py`; judge scored 16/16 on the hand-picked pairs
- Found and fixed a real bug: the judge only saw the top-1 similar memory, so a genuine update was missed; now judges the top 3 candidates
- Retrieval: minimum-similarity cutoff (0.15) after the smoke test showed unrelated memories surfacing
- Loud LLM failures: an API failure now returns an error through categorize, pipeline and judge instead of silently skipping, with tests
- `docs/scoring_notes.md` with measured results; test suite now at 81 tests

**Evaluation harness (PR #7)**
- `evaluation/locomo_utils.py` and `run_eval.py`: LoCoMo evaluation of adaptive memory vs truncation (10 and 40 turns) vs single summary
- Metrics: token F1, LLM-judged correctness, approximate context tokens, evidence recall at 3, 5 and 10
- Resumable results, checkpoint-order guard, ingest log / database consistency check, `--fresh`, `--dry-run` (full flow offline)
- Groq rate-limit handling: request and token pacing, retry-after parsing, daily-limit detection that stops cleanly and resumes
- Methods compared only on the questions every method answered
- `summarize_results.py`: tables, CSV and 6 charts (accuracy, F1, context tokens, accuracy vs cost, by question type, retrieval recall)
- `retrieval_analysis.py` and `retrieval_experiments.py`: retrieval-only analysis and 5 ranking strategies, no LLM calls
- `contradiction_eval.py`: 60 labelled pairs with a free gate evaluation and a threshold sweep

**Measured findings so far (conv-26, first 100 turns)**
- Adaptive memory gives the model about 250 context tokens vs about 1,850 for truncation-40
- Only 32 of 100 turns were stored (65 skipped as recent_chat filler); evidence for 61% of questions was still in memory
- Of the questions missed at top-5: 16 never stored, 9 stored but ranked too low; keyword, hybrid and embedding ranking are within noise
- Contradiction gate: 92% of replace-worthy pairs reach the judge, 0 of 15 unrelated pairs do

**Embeddings (PR #7)**
- Diagnosed the Hugging Face 402 (free credits ran out) and moved embeddings to a local fastembed model (same MiniLM),
  with the HF API kept as a fallback; no token or rate limit

**Chrome extension (PR #7)**
- Popup rewritten as a demo panel: server status, add message with ADDED / REPLACED / SKIP result, live memory list,
  retrieve, copy as prompt, copy JSON snapshot, clear
- Background worker queue (one request at a time with a gap) so opening a chat doesn't trip the rate limit
- Content script captures the user's messages only and waits for the text to settle; popup uses `textContent` only (no HTML injection)
- Tested in headless Chromium against the real server code

**Setting to test the filler hypothesis (PR #8)**
- Optional `STORE_RECENT_CHAT` (off by default) with tests

### To do

**T1. Check the ChatGPT selector** (Saturday night, 5 min, no Groq)
- Reload the extension in `chrome://extensions`, open chatgpt.com, send any message, press F12 and run in the Console:
  `document.querySelectorAll('[data-message-author-role="user"]').length`
- Done when: the number is greater than 0. If it is 0, send the DOM snippet to Claude and fix `SITE_SELECTORS` in `extension/content.js`.
- Also check: with the server running, send a real message in ChatGPT and confirm it appears in the popup's memory list.

**T2. Retrieval tuning** (Saturday night, free, with Claude)
- Make the ranking weights configurable (currently fixed at 0.7 similarity + 0.3 importance in `retrieve.py`).
- Run `python evaluation/retrieval_analysis.py --conv conv-26 --max-turns 100` for similarity weights 0.7, 0.9 and 1.0,
  and `retrieval_experiments.py` for top-k 5, 8 and 10.
- Done when: a table of recall@5 and recall@10 per setting is added to `docs/scoring_notes.md` and the chosen defaults are in `retrieve.py`
  (with tests updated). Keep the old weights if nothing beats them by a clear margin; say so honestly.

**T3. Extension: "Insert into chat box" button** (Saturday night or Sunday morning)
- Add a button in the popup that sends a message to the content script; the content script finds ChatGPT's prompt box
  (a `contenteditable` element, `#prompt-textarea`), fills it with the memory prompt (same text as "Copy as prompt") and fires an
  `input` event so the page notices.
- If the prompt box can't be found, fall back to copying to the clipboard and show "Copied instead".
- Done when: in a fresh ChatGPT chat the button fills the prompt box and you can press Enter. Test twice.
  This feature depends on ChatGPT's page structure, so keep Copy as prompt as the backup in the demo.

**T4. Sunday Groq runs** (after the quota resets; run in this order and paste each output to Claude)
- [ ] 1. Finish the summary baseline (~15k tokens):
      `python evaluation/run_eval.py --convs conv-26 --max-turns 100 --methods summary --max-questions 20`
- [ ] 2. Contradiction judge (~30k tokens): `python evaluation/contradiction_eval.py --judge`
- [ ] 3. Adaptive with filler kept (~70k to 100k tokens):
      `set STORE_RECENT_CHAT=1` then
      `python evaluation/run_eval.py --convs conv-26 --max-turns 100 --methods adaptive --max-questions 20 --out evaluation/results_keep`
      then close that Command Prompt window so the setting does not leak into the demo
- Done when: all 20 questions have an answer for adaptive, truncation-10, truncation-40 and summary, and the judge results are saved.
  Add the numbers to the PR descriptions (store-recent-chat, contradiction eval).

**T5. Merge results and freeze charts** (Sunday afternoon)
- Copy Vrishti's `conv-30__*.jsonl` files into `evaluation/results/t100/`.
- Run `python evaluation/summarize_results.py`. Check that every method shows the same n (the script only compares questions all methods answered).
- Copy the final PNGs and CSVs from `evaluation/figures/` into a committed folder `docs/figures/` so Peehu can use them.
- Done when: `docs/figures/` has the final charts and `summary.csv`, and the numbers match what Claude read out.

**T6. README rewrite** (Sunday evening, with Claude)
- Remove the Hugging Face token steps, say embeddings are local. Describe the real architecture and the extension panel.
- How to run: server, extension, tests, evaluation (the commands), the demo.
- A results section with the final numbers and the honest limitations.
- Done when: a person who has never seen the repo can follow it to run the server and the extension.

**T7. Results note for Peehu: `docs/results_for_slides.md`** (Sunday evening, with Claude)
- Final numbers in a table, one line per chart saying what it shows, the 4 or 5 findings in plain words,
  the limitations list, the commands to reproduce.
- Done when: Peehu can build every results slide from this file alone.

**T8. Demo rehearsal and backup** (Sunday night, then Monday morning)
- Pre-flight checklist: server starts (`uvicorn server:app --port 8000`), `.env` has a working Groq key, extension reloaded,
  popup shows "server connected", `STORE_RECENT_CHAT` is NOT set (`echo %STORE_RECENT_CHAT%` prints nothing), the conversation is cleared.
- Demo script (about 60 seconds):
  1. Popup shows "server connected"
  2. Add "I live in Delhi and work as a data analyst." → ADDED
  3. Add "I moved to Mumbai last month." → REPLACED, only Mumbai remains
  4. Ask "where does the user live?" → Retrieve returns only the relevant memory
  5. Copy as prompt (or Insert into chat box) → paste into a new ChatGPT chat
- Run the whole script twice. Record a 1 to 2 minute screen video of it as a backup in case something fails live.
- Keep at least 20k Groq tokens unused for Monday (check by not running anything on Monday morning).
- Done when: the demo ran twice without a surprise and the video exists.

**T9. Monday: final check** (morning)
- Check every number in the slides against `docs/results_for_slides.md`; check that the repo's `main` has everything and the README works.
- Submit.

---

## Vrishti — evaluation and EDA

### Already done (Oct 6, `evaluation/eda_starter.ipynb`)
- LoCoMo loaded and profiled: 5,882 turns, 10 conversations, turns per conversation, turns per speaker, missing values, duplicate dialogue ids
- `clean_turns()` adds session timestamps (0 rows removed)
- Truncation baseline and a hierarchical single-summary baseline (chunked, because 419 turns exceed the model limit)
- 20-turn adaptive pilot on conv-26 (8 stored, 12 skipped as recent_chat)
- 20 hand-labelled contradiction pairs from real LoCoMo turns, with precision / recall / F1 per label (95% accuracy)
- QA exploration and evidence coverage by checkpoint (50 to 400 turns) for truncation vs adaptive
- Findings and limitations section, three charts

### To do

**V1. Setup** (Saturday night, 30 min, do this first)
- Pull `main`, create a new venv, `pip install -r requirements.txt` (includes fastembed; the first run downloads a ~90 MB model).
- Put your own Groq key in `.env` and set `GROQ_MODEL=openai/gpt-oss-120b`. Keep `data/locomo10.json` in place.
- Done when: `python -m pytest -q` passes (81 tests) and `python evaluation/run_eval.py --dry-run --convs conv-26 --max-turns 100 --methods adaptive --max-questions 5` finishes.

**V2. One more conversation: conv-30** (Sunday, uses YOUR Groq quota; about 150k tokens, so nothing else on your key that day)
- Run in this order. If a command prints `STOPPED: Groq's limit...`, rerun the same command later; it resumes.
  1. `python evaluation/run_eval.py --convs conv-30 --max-turns 100 --methods adaptive --max-questions 15`
  2. `python evaluation/run_eval.py --convs conv-30 --max-turns 100 --methods truncation --max-questions 15`
  3. `python evaluation/run_eval.py --convs conv-30 --max-turns 100 --methods summary --max-questions 15`
- Send Tvisha the files `evaluation/results/t100/conv-30__*.jsonl` (results are not in git on purpose).
- Done when: those files hold 15 answers for each of adaptive, truncation-10, truncation-40 and summary. Optional if quota remains: conv-48 the same way.

**V3. EDA notebook: finish and fix** (Saturday night and Sunday morning, no Groq; this is the core EDA deliverable for the course)
- [ ] Questions by category (multi-hop, temporal, open-domain, single-hop, adversarial) as a bar chart. Use `answerable_questions`
      from `evaluation/locomo_utils.py` for the analysis (it drops adversarial questions that have no gold answer) and state how many were dropped.
- [ ] How far back the evidence is: for each question, the distance between its evidence turn and the end of the conversation; histogram.
- [ ] Data-quality notes: malformed evidence ids (e.g. "D8:6; D9:17"), image-only turns, any missing fields; one cleaning step that handles them
      (see `evidence_ids` in `locomo_utils.py` for the regex).
- [ ] Fix the coverage metric in the pilot: count only questions whose evidence lies inside the turns processed.
      Counting all 199 questions after 20 turns makes the result look like 3%.
- [ ] Export 3 to 4 clean figures (readable titles and axis labels, no leftover default styling) to `evaluation/figures/eda_*.png`.
- Done when: the notebook runs top to bottom, has a short written finding under every chart, and the PNGs exist.

**V4. Contradiction test set** (Saturday night and Sunday, no Groq)
- [ ] Add 20 to 30 harder pairs to your set. Your current 20 are 15 consistent, 3 unrelated, 1 contradicts and 1 updates, so the 95% mostly
      measures the easy class. Aim for at least 10 real updates and 10 contradictions: implicit updates, partial updates, different person,
      past vs present, numeric changes. Use real LoCoMo turns where possible.
- [ ] Independent labelling: Tvisha sends you a CSV of the 60 pairs from `evaluation/contradiction_eval.py` without labels. Label each as
      updates / contradicts / consistent / unrelated without looking at the code, then compare with Tvisha's labels.
      Report the agreement (% of pairs) and list the disagreements.
- Done when: the notebook has the extended set, per-label precision and recall, and a short agreement table.

---

## Peehu — slides and report text (no code)

### To do

**P1. Slide skeleton** (Saturday night or Sunday morning, no results needed yet)
- 10 to 12 slides, suggested order:
  1. Title and team
  2. Problem: long chats get truncated or squashed into one summary, so memory is lost or goes stale
  3. Idea: categorize, score, detect contradictions, retrieve only what is relevant, export a model-agnostic snapshot
  4. Architecture diagram (extension → FastAPI server → memory engine → SQLite)
  5. How a message is processed (categorize → score → contradiction check → store)
  6. Live demo (see Tvisha's demo script) or the backup video
  7. Evaluation setup: LoCoMo, the three methods, the metrics
  8. Results: accuracy and context tokens (charts from `docs/figures/`)
  9. What we found: where the memory fails (retrieval ranking, what got stored), contradiction results
  10. EDA highlights (Vrishti's figures)
  11. Limitations (honest, see below)
  12. Future work and conclusion

**P2. Fill in the results slides** (Sunday evening, after `docs/results_for_slides.md` and `docs/figures/` exist)
- Use only the numbers in that file. Do not round up or reword them into claims the data doesn't support.
- 3 or 4 screenshots of the extension popup (or frames from Tvisha's demo video).

**P3. Limitations slide, kept honest**
- One main conversation plus one more, and a small number of questions; the same model answers and judges; contradiction pairs written by the
  team (the agreement check helps); LoCoMo is two people chatting, not a user talking to an assistant; truncation and summary baselines are simple versions.

**P4. Speaker notes and rehearsal** (Sunday night)
- 20 to 30 seconds of speaker notes per slide; one run-through with the demo and a timer.

**P5. Review** (Sunday night)
- Send the draft to Tvisha by Sunday night; Tvisha checks every number against the results note. Fix, then final copy on Monday morning.

---

## If things slip (plan B)
- **Groq quota not back or a run fails:** report what exists. The conv-26 results (adaptive 20 questions, truncation, partial summary) are real;
  say how many questions each number is based on. Do not hide a smaller sample.
- **Vrishti's conv-30 results don't arrive:** the report uses conv-26 only and says so; her EDA and contradiction work still count.
- **The extension breaks during the demo:** play the backup video, or run the same steps through `http://localhost:8000/docs`.

## Cut / future work (say so in the report)
- Claude.ai and Gemini selectors (ChatGPT only)
- Recency decay, SQLite search optimisation (no evidence they are needed)
- LongMemEval / ConvoMem (LoCoMo only)
- Automatic cross-model import (done manually via "Copy as prompt")

## Timeline
| When | Tvisha | Vrishti | Peehu |
|---|---|---|---|
| Sat night | T1 selector check, T2 tuning, T3 insert button | V1 setup, V3 EDA, V4 pairs | P1 slide skeleton |
| Sun morning to afternoon | T4 Groq runs | V2 conv-30 runs, V3, V4 | Skeleton and design |
| Sun evening | T5 merge and freeze charts, T6 README, T7 results note | Send results to Tvisha; finish EDA figures | P2 fill in results |
| Sun night | T8 demo rehearsal and backup video | Final notebook | P3, P4, P5, send draft |
| Mon | T9 final check, demo, submit | Standby | Final copy |