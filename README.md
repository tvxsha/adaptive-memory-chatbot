# Adaptive Memory for AI Chatbots

An adaptive memory management system for long conversations with AI chatbots.
Instead of truncating old messages or squashing everything into one summary,
this system **categorizes** conversational data (facts, preferences, goals,
decisions, completed tasks), **scores** each item by likely future usefulness,
**detects contradictions** between new and old information, and **retrieves**
only what's relevant to the current query — then exports a model-agnostic
memory snapshot that can move between ChatGPT, Claude, and Gemini.

Built for: *Adaptive Memory for AI Chatbots* — EDA Project Review 0
Team: Peehu Gupta, Tvisha Thakur, Vrishti Sharma

## Architecture

```
Chrome Extension (JS)  --HTTP-->  FastAPI server (Python)  -->  Memory Engine
   captures chat turns              /add_message                categorize
   injects retrieved memory         /retrieve                   score
   shows live memory panel          /export                     detect contradictions
                                                                  store (SQLite)
                                                                  retrieve (embeddings)
```

- **`memory_engine/`** — the core Python logic: categorization, scoring,
  contradiction detection, storage, and retrieval. This is what you evaluate
  and write your EDA/report around.
- **`server.py`** — a small FastAPI server that exposes the memory engine over
  HTTP so the Chrome extension can talk to it.
- **`extension/`** — the Chrome extension (Manifest V3) skeleton: background
  service worker, a content script placeholder (you'll fill in site-specific
  selectors per chatbot), and a popup panel.
- **`evaluation/`** — where the EDA/benchmarking work lives: loading
  LoCoMo/LongMemEval/ConvoMem, cleaning, computing metrics, and plotting
  results against the truncation and single-summary baselines.

## Setup

### 1. Prerequisites
- Python 3.10+ (3.11 or 3.12 recommended for smoothest package installs — very
  new Python releases, e.g. 3.14, may lack prebuilt wheels for numpy/pandas)
- Google Chrome (for loading the unpacked extension)
- A free [Groq API key](https://console.groq.com/keys)
- A free [Hugging Face token](https://huggingface.co/settings/tokens) (used
  for embeddings via the HF Inference API — no local PyTorch needed)

### 2. Clone and set up a virtual environment

```bash
git clone https://github.com/<your-username>/adaptive-memory-chatbot.git
cd adaptive-memory-chatbot
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and paste your Groq key and Hugging Face token:

```
GROQ_API_KEY=your_key_here
HF_TOKEN=your_hf_token_here
```

### 4. Run the backend server

```bash
uvicorn server:app --reload --port 8000
```

You should see the server come up at `http://localhost:8000`. Visit
`http://localhost:8000/docs` for the interactive API docs (FastAPI generates
this automatically) — useful for testing `/add_message` and `/retrieve`
without the extension.

### 5. Load the Chrome extension

1. Go to `chrome://extensions`
2. Enable **Developer mode** (top-right toggle)
3. Click **Load unpacked**
4. Select the `extension/` folder
5. The extension icon should appear in your toolbar

The content script (`extension/content.js`) currently has placeholder
selectors — you'll need to inspect each chatbot's page (ChatGPT, Claude,
Gemini) and fill in the actual DOM selectors for message containers, since
each site's HTML differs. This is the main "glue" work for a working demo.

### 6. Run a quick manual test

With the server running:

```bash
curl -X POST http://localhost:8000/add_message \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "test1", "role": "user", "text": "My favorite color is blue"}'

curl "http://localhost:8000/retrieve?conversation_id=test1&query=what color do I like"
```

### 7. Run the test suite

```bash
pytest tests/
```

### 8. Evaluation / EDA notebook

```bash
jupyter notebook evaluation/eda_starter.ipynb
```

This is where you'll load LoCoMo/LongMemEval/ConvoMem, clean them, run your
memory pipeline against the truncation and single-summary baselines, and
produce the comparison charts (token usage, consistency, contradiction
handling) for your report.

## Project status

This is a starter scaffold — the categorization, scoring, and contradiction
logic use working baseline implementations (heuristics + Groq LLM calls) that
you should refine and tune as you go. See inline `# TODO` comments in
`memory_engine/` for the main places to extend.

## Setting up the GitHub repo

See `GITHUB_SETUP.md` for step-by-step instructions on creating the repo and
pushing this code for the first time.
