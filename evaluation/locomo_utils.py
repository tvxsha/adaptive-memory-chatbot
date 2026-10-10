"""Pure helpers for loading LoCoMo and scoring answers.

No network and no LLM calls in this file, so everything here is unit-tested
offline (tests/test_eval_utils.py).
"""
import json
import re
import string
from collections import Counter

# LoCoMo question categories (naming follows the LoCoMo paper / common usage).
# Category 5 ("adversarial") questions have no gold answer, so accuracy is
# only reported for categories 1-4.
CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

_EVIDENCE_ID = re.compile(r"D\d+:\d+")


def load_locomo(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def conversation_turns(entry: dict, include_captions: bool = True) -> list[dict]:
    """Flat, chronological list of turns for one LoCoMo conversation.

    Each turn: {"dia_id", "speaker", "text", "session", "date"}.
    Turns with no text are dropped. If a turn shared a photo and
    include_captions is True, the photo caption is appended to the text,
    because many questions ask about what was in the photo.
    """
    conv = entry["conversation"]
    session_keys = sorted(
        (k for k in conv if k.startswith("session_") and not k.endswith("_date_time")),
        key=lambda k: int(k.split("_")[1]),
    )
    turns = []
    for key in session_keys:
        date = conv.get(f"{key}_date_time", "")
        for t in conv[key]:
            text = (t.get("text") or "").strip()
            caption = (t.get("blip_caption") or "").strip()
            if include_captions and caption:
                text = f"{text} (shared a photo: {caption})".strip()
            if not text:
                continue
            turns.append({
                "dia_id": t.get("dia_id"),
                "speaker": t.get("speaker", ""),
                "text": text,
                "session": key,
                "date": date,
            })
    return turns


def format_turn(turn: dict) -> str:
    """The exact string stored in memory for a turn.

    The memory engine stores plain text only (no dates or speakers), so the
    date and speaker are put into the text itself. Without this, temporal
    questions ("When did Caroline go ...?") and "who said it" questions cannot
    be answered from memory at all.
    """
    return f"[{turn['date']}] {turn['speaker']}: {turn['text']}"


def evidence_ids(evidence) -> list[str]:
    """Normalizes a question's evidence list. The dataset sometimes has
    entries like 'D8:6; D9:17' or stray text, so extract ids with a regex."""
    ids = []
    for item in evidence or []:
        ids.extend(_EVIDENCE_ID.findall(str(item)))
    return ids


def answerable_questions(entry: dict) -> list[dict]:
    """Questions that have a gold answer (drops category 5), each with its
    original position so results can be matched back."""
    out = []
    for idx, q in enumerate(entry["qa"]):
        if q.get("category") == 5 or "answer" not in q:
            continue
        out.append({
            "idx": idx,
            "question": q["question"],
            "answer": str(q["answer"]),
            "category": q["category"],
            "evidence": evidence_ids(q.get("evidence")),
        })
    return out

def questions_within(questions: list[dict], turns: list[dict]) -> list[dict]:
    """Keeps only questions whose evidence turns all lie inside `turns`.
    Used when evaluating a conversation prefix ("the first 100 turns"): a
    question about something said later cannot be answered yet."""
    ids = {t["dia_id"] for t in turns}
    return [q for q in questions if q["evidence"] and all(e in ids for e in q["evidence"])]

# ---- answer scoring (no LLM) -------------------------------------------
def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def approx_tokens(text: str) -> int:
    """Rough token count (words * 1.3). Good enough to compare methods; not a
    tokenizer-exact number, so label it 'approx' in any chart."""
    return int(len(text.split()) * 1.3)


# ---- evidence checks ----------------------------------------------------
def evidence_texts(turns_by_id: dict, ids: list[str]) -> list[str]:
    """Formatted memory strings for a question's evidence turns (skipping ids
    that don't exist in the conversation)."""
    return [format_turn(turns_by_id[i]) for i in ids if i in turns_by_id]


def evidence_hit(ev_texts: list[str], retrieved: list[str], require_all: bool) -> bool:
    if not ev_texts:
        return False
    found = [t in set(retrieved) for t in ev_texts]
    return all(found) if require_all else any(found)