# Adaptive Memory for AI Chatbots

A memory layer for long chatbot conversations. Instead of cutting off old messages
or squashing the whole chat into one summary, it decides what is worth remembering,
keeps it up to date, and gives the chatbot only the few memories that matter for the
current question.

Course project for BACSE301 (Exploratory Data Analysis), VIT.
Team: Peehu Gupta, Tvisha Thakur, Vrishti Sharma.

## How it works

```
Chrome extension  --HTTP-->  FastAPI server  -->  Memory engine
 reads chat turns            /add_message          1. categorize (6 categories)
 injects memories            /retrieve             2. score (how useful later)
 shows a memory panel        /export               3. check contradictions
                                                   4. store (SQLite)
                                                   5. retrieve (embeddings)
```

1. **Categorize.** Each message becomes a fact, preference, goal, decision,
   completed task, or recent chat. Small talk is not stored.
2. **Score.** Each memory gets an importance score from its category and how
   specific it is. An optional trained model can add to this (see below).
3. **Check contradictions.** A new memory is compared with similar old ones
   (cosine similarity gate 0.35). A language model judge decides whether the new
   one replaces the old one or both are kept.
4. **Store.** Memories live in a local SQLite database.
5. **Retrieve.** For a question, the top 5 memories are chosen using
   0.7 x similarity + 0.3 x importance, and only those are sent to the chatbot.

Embeddings run locally (all-MiniLM-L6-v2 through fastembed), so there is no
embedding API cost. The language model is Groq `openai/gpt-oss-120b`.

## Results

Evaluated on LoCoMo (long conversations with questions about them). 77 questions
from 4 conversations (conv-26, conv-30, conv-41, conv-42), first 100 turns each.

| Method | Judged accuracy | Context tokens per question |
|---|---|---|
| **Adaptive memory (ours)** | 0.494 | **252** |
| Single summary | **0.532** | 598 |
| Truncation, last 40 turns | 0.247 | 1,724 |
| Truncation, last 10 turns | 0.091 | 456 |

- Clearly better than truncation (p = 0.0013 against the last 40 turns).
- Statistically tied with the single summary on accuracy (difference -0.04, p = 0.72)
  while using less than half the context, about 2.2 times more accuracy per 1,000 tokens.
- Weak spot: multi-hop questions (0.12 against 0.62 for the summary, 8 questions).
  The same model answers and judges, so treat these as indicative.

### Trained models

| Model | What it does | Result |
|---|---|---|
| Importance model (Gradient Boosting) | Predicts which turns will be needed later | ROC AUC 0.785 (leave one conversation out) vs 0.514 for the hand-set score. Retrieval improves by about 4 points (all evidence in top 5). Off by default. |
| Contradiction classifier (distilroberta, fine-tuned) | Replace the old memory or keep both | On 148 pairs labelled independently by three people: precision 0.92, recall 0.77. The language model judge scores 0.97 and 0.83. Evaluation only, not wired into the server. |
| Categorizer (Logistic Regression on MiniLM) | Files a message into a category | On 80 test messages: accuracy 0.55, macro F1 0.51, against 0.61 and 0.56 for the language model. Evaluation only. |

Contradiction labels were made independently by three annotators (Fleiss kappa 0.775
on four classes, 0.848 on replace or keep). The categorizer labels were made by hand
by the three team members, with agreement of kappa 0.38 on the shared messages, so the
categories partly overlap.

An ablation shows that missing information (never stored or wrongly replaced) is a
bigger loss than ranking it badly.

## Setup

Requirements: Python 3.10 to 3.12, Google Chrome, a free
[Groq API key](https://console.groq.com/keys).

```bash
git clone https://github.com/tvxsha/adaptive-memory-chatbot.git
cd adaptive-memory-chatbot
python -m venv venv
venv\Scripts\activate          # Mac or Linux: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env           # Windows: copy .env.example .env
```

Open `.env` and set `GROQ_API_KEY`. Optional settings:

- `STORE_RECENT_CHAT=1` keeps small talk (off by default).
- `USE_LEARNED_IMPORTANCE=1` adds the trained importance model to the score.
- `HF_TOKEN` is only needed if fastembed is not installed.

## Run the server and extension

```bash
uvicorn server:app --reload --port 8000
```

Check it at `http://localhost:8000/docs`. Then in Chrome open `chrome://extensions`,
turn on Developer mode, click Load unpacked, and pick the `extension/` folder.

Quick test without the extension:

```bash
curl -X POST http://localhost:8000/add_message -H "Content-Type: application/json" -d "{\"conversation_id\": \"test1\", \"role\": \"user\", \"text\": \"My favorite color is blue\"}"
curl "http://localhost:8000/retrieve?conversation_id=test1&query=what color do I like"
```

## Tests

```bash
python -m pytest
```

86 tests, none of them need a Groq key.

## Reproduce the evaluation

Download LoCoMo and save it as `data/locomo10.json`. Groq's free tier allows about
200k tokens a day, so the main run stops cleanly when the limit is hit; run the same
command again the next day and it resumes.

```bash
python evaluation/run_eval.py --convs conv-26 --max-turns 100 --max-questions 20
python evaluation/summarize_results.py          # tables and charts
python evaluation/significance.py               # confidence intervals and p-values
python evaluation/ablation_retrieval.py         # what each part contributes
python evaluation/retrieval_analysis.py         # where retrieval fails
```

Trained models:

```bash
python evaluation/train_importance.py                  # importance model
python evaluation/train_text_importance.py             # version the live system can use
python evaluation/importance_nested_weight.py          # effect on retrieval
python evaluation/train_contradiction.py --test-csv evaluation/contradiction_majority.csv
python evaluation/judge_on_labels.py --csv evaluation/contradiction_majority.csv
python evaluation/train_categorizer.py                 # add --llm to compare with the language model
```

The fine-tuned contradiction models are trained in
`evaluation/colab_contradiction_finetune.ipynb` (Colab GPU, about 3 minutes each).

## Folder guide

| Folder or file | What is in it |
|---|---|
| `memory_engine/` | categorize, score, contradiction, store, retrieve, embeddings |
| `server.py` | FastAPI server |
| `extension/` | Chrome extension (Manifest V3) |
| `evaluation/` | all experiments, label sheets, and training scripts |
| `docs/figures/` | result tables and charts used in the report |
| `tests/` | unit tests |

## Limits

- Small sample: 77 questions from 4 conversations, so the difference from the summary
  baseline is not statistically significant either way.
- The same model answers and judges.
- Multi-hop questions do not work well yet.
- The fine-tuned contradiction classifier and the categorizer are evaluation results.
  The live server uses the language model for both.