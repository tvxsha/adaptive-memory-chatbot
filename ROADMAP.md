# Roadmap — Adaptive Memory for AI Chatbots

Team: Tvisha Thakur (backend/systems), Peehu Gupta (extension/frontend), Vrishti Sharma (evaluation/EDA)

This roadmap splits work into three parallel tracks so all three of you can work
simultaneously without blocking each other for most of the timeline. Each
person's section is ordered by priority, and every task is marked:

- 🟢 **CAN START NOW** — no dependency, begin immediately
- 🔴 **BLOCKED** — waiting on something specific (named explicitly)

---

## Phase 0 — Setup (everyone, do this first)

- [ ] 🟢 Clone the repo, create your own venv, install `requirements.txt`
- [ ] 🟢 Get your own Groq API key and Hugging Face token, add to your local `.env`
- [ ] 🟢 Confirm `uvicorn server:app --reload --port 8000` runs and `/health` returns `{"status": "ok"}`
- [ ] 🟢 Read through `README.md` and skim every file in `memory_engine/` once, so everyone has the same mental model before splitting off

---

## Track 1 — Tvisha: Core memory engine & backend

### Can start now 🟢
- [ ] Finish validating the full pipeline manually via `/docs`: run 5–10 varied
      test messages through `/add_message` and confirm categorization looks
      reasonable (facts vs preferences vs goals vs recent_chat)
- [ ] Write 10–15 more test cases in `tests/test_memory_engine.py` covering
      edge cases: empty string, very long message, message with no clear
      category, duplicate message sent twice
- [ ] Review `memory_engine/score.py`'s `CATEGORY_WEIGHTS` heuristic — these
      are placeholder numbers, start a doc/notes file listing what you think
      they should be and why (informal, just to prep for Phase 2 tuning)
- [ ] Add basic logging (Python `logging` module) throughout
      `memory_engine/pipeline.py` so every step (categorize → score →
      contradiction check → store) prints what happened — needed for
      debugging once real conversations start flowing through
- [ ] Set up the GitHub repo per `GITHUB_SETUP.md`, add Peehu and Vrishti as
      collaborators, create the `main` branch structure

### Blocked 🔴
- [ ] **Tune `CATEGORY_WEIGHTS` and `SIMILARITY_THRESHOLD` with real numbers**
      — blocked on Vrishti's evaluation harness existing (Phase 2) so there's
      something to measure improvement against
- [ ] **Add recency decay to scoring** (`recency_decay()` already stubbed in
      `score.py`) — blocked on evaluation results showing pure importance
      scoring is losing too much temporal signal; don't build this
      speculatively
- [ ] **Optimize the SQLite similarity search** (currently loads all items
      into memory and does cosine similarity in Python) — blocked on Vrishti's
      benchmark runs showing this is actually a bottleneck at real dataset
      scale; premature to optimize now

---

## Track 2 — Peehu: Chrome extension & cross-model integration

### Can start now 🟢
- [ ] Inspect ChatGPT's DOM (chat.openai.com / chatgpt.com) in devtools and
      confirm/update the message container selector in
      `extension/content.js` (`SITE_SELECTORS`) — this one is closest to
      correct already, verify it
- [ ] Inspect Claude.ai's DOM and fill in the real selector (currently a
      `TODO` placeholder) — message containers, and how to tell user vs
      assistant turns apart
- [ ] Inspect Gemini's DOM and do the same
- [ ] Load the extension unpacked (`chrome://extensions` → Developer mode →
      Load unpacked) and confirm it appears in the toolbar with no manifest
      errors — this doesn't need the server running yet
- [ ] Improve `extension/popup.html` / `popup.js` UI — right now it's a bare
      list; make it visually match a "memory panel" (grouped by category,
      maybe collapsible sections)
- [ ] Draft the JSON schema for the **model-agnostic memory snapshot** export
      format mentioned in the abstract — `store.py`'s `export_snapshot()`
      already returns a basic version; decide if it needs more fields
      (e.g. source model, export timestamp format, versioning) for it to be
      genuinely re-importable into a different chatbot

### Blocked 🔴
- [ ] **End-to-end test: extension captures a real ChatGPT conversation and
      injects retrieved memory into a new session** — blocked on your
      selector work above being done AND on the server running locally on
      whoever's testing machine (coordinate with Tvisha to test together)
- [ ] **Build the "inject memory into prompt" feature** (taking `/retrieve`
      results and actually inserting them into the chat input before the
      user sends a message) — blocked on the popup UI and selectors being
      stable first; no point wiring injection to a selector that might change
- [ ] **Cross-model import** (taking an exported snapshot from ChatGPT and
      loading it into a Claude conversation) — blocked on the snapshot schema
      being finalized (your task above) and on selectors for at least two
      sites being done

---

## Track 3 — Vrishti: Evaluation, EDA & benchmarking

### Can start now 🟢
- [ ] Download LoCoMo and LongMemEval datasets (check their GitHub repos /
      Hugging Face dataset pages) into `data/` — this needs no code, just
      getting the files
- [ ] Do initial exploratory profiling of the raw datasets in
      `evaluation/eda_starter.ipynb`: conversation length distribution,
      number of turns, any missing fields — this is pure pandas/numpy work
      and doesn't touch `memory_engine` at all
- [ ] **Start building the hand-crafted 20–30 conversation contradiction test
      set** mentioned in the methodology slide — this is just writing
      example conversations with known contradictions (e.g. "I live in
      Delhi" ... later ... "I moved to Mumbai") and labeling the correct
      relation (contradicts/updates/consistent/unrelated) in a CSV or JSON.
      Pure data-authoring work, no dependencies.
- [ ] Write the data cleaning functions properly (the notebook currently has
      a placeholder `clean_turns()`) — dedup logic, handling missing
      timestamps, normalizing text — using the real LoCoMo/LongMemEval schema
      once downloaded
- [ ] Implement the **truncation baseline** and **single-summary baseline**
      properly (currently stubbed in the notebook) — the summary baseline
      needs one Groq call per conversation to condense it, which you can
      write and test independently of the rest of the pipeline

### Blocked 🔴
- [ ] **Run the full evaluation comparing adaptive memory vs. truncation vs.
      summary on token usage, consistency, factual accuracy** — blocked on
      Tvisha's pipeline being stable (it is, as of last test) AND on your own
      baseline implementations above being done first
- [ ] **Run precision/recall on contradiction detection** — blocked on your
      hand-crafted test set (above) being finished
- [ ] **Produce final comparison charts for the report** (token usage bar
      chart, consistency-vs-conversation-length line chart, category
      distribution) — blocked on the full evaluation run above actually
      producing numbers to plot
- [ ] **Tune `SIMILARITY_THRESHOLD` in `contradiction.py` and
      `CATEGORY_WEIGHTS` in `score.py`** — this is listed under Tvisha too;
      you two should do this together once your contradiction test set
      exists, since it's the ground truth for tuning

---

## Phase 2 — Integration (all three, once individual tracks converge)

- [ ] 🔴 Full end-to-end demo: real chatbot conversation → extension captures
      it → memory engine processes it → popup shows live categorization →
      export snapshot works. Blocked on Track 1 (stable backend) + Track 2
      (working selectors) both being done.
- [ ] 🔴 Run the complete evaluation suite and freeze the results for the
      report. Blocked on Track 3's evaluation work being complete.
- [ ] 🔴 Write up the report/slides using the frozen results. Blocked on the
      above.

## Suggested order of operations

1. **This week:** everyone works their "can start now" list in parallel —
   these don't block each other at all.
2. **Once Tvisha's tests + logging are solid and Vrishti's baselines exist:**
   start the tuning loop (Tvisha + Vrishti together).
3. **Once Peehu has at least one site's selectors working:** do a joint
   test session (all three) running a real conversation through the whole
   system.
4. **Last:** full evaluation run, charts, report.

## Notes

- If anyone finishes their "can start now" list early, the next best use of
  time is picking up a task from Phase 2 prep (e.g. Peehu could start
  drafting demo-day conversation scripts; Tvisha could start on the
  scoring-tuning notes; Vrishti could expand the contradiction test set
  beyond 30 examples).
- Flag it in the group chat immediately if a "blocked" task's dependency
  finishes early — don't wait for a scheduled check-in to unblock each other.