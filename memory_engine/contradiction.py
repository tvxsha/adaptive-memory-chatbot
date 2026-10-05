"""Detects whether a new statement contradicts an existing memory item.

Two-stage approach:
1. Cheap filter — only compare the new item against memories that are
   semantically similar enough to plausibly be about the same thing
   (cosine similarity above a threshold). Avoids an LLM call per item.
2. LLM judgment — for similar-enough candidates, ask the model whether the
   new statement contradicts, updates, or is consistent with the old one.

TODO: your hand-built 20-30 conversation contradiction test set is exactly
what should drive tuning SIMILARITY_THRESHOLD and the prompt below.
"""
import json
import logging
from groq import Groq

from memory_engine.config import GROQ_API_KEY, GROQ_MODEL
from memory_engine.embeddings import embed, cosine_similarity

# Provisional: lowered from 0.55 after check_similarity.py showed real updates
# (e.g. "I live in Delhi" -> "I just moved to Mumbai") scoring 0.39-0.54.
# Re-tune with the full contradiction test set.
SIMILARITY_THRESHOLD = 0.35

# Above this, a new statement is treated as a repeat of an existing memory
# (skipped) rather than sent to the LLM for a contradiction judgment.
DUPLICATE_THRESHOLD = 0.97
# How many of the most similar memories get sent to the LLM judge per new
# message. Each one is a separate Groq call, so this trades accuracy for
# rate-limit headroom.
MAX_CANDIDATES_TO_JUDGE = 3
_client = None
logger = logging.getLogger("memory_engine.contradiction")

def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


CONTRADICTION_PROMPT = """Compare these two statements from the same person's \
conversation history.

Old statement: "{old_text}"
New statement: "{new_text}"

Respond with ONLY a JSON object:
{{"relation": "<contradicts|updates|consistent|unrelated>", "reasoning": "<one short sentence>"}}

- "contradicts": the new statement directly conflicts with the old one; both cannot be true
- "updates": the old statement is NO LONGER TRUE because something changed (moved cities, quit a job, rescheduled). The new statement should replace the old one
- "consistent": the old statement is STILL TRUE. This includes when the new statement only adds detail or elaborates on it (e.g. "I have a dog" followed by "I have a golden retriever named Max"), or is about a different person
- "unrelated": they aren't about the same topic

If the old statement could still be true after reading the new one, answer "consistent", not "updates".
"""

def find_candidates(new_text: str, existing_items: list[dict]) -> list[dict]:
    """existing_items: list of {"id", "text", "embedding"}.
    Returns the subset above the similarity threshold, most similar first.
    """
    new_emb = embed(new_text)
    scored = []
    for item in existing_items:
        sim = cosine_similarity(new_emb, item["embedding"])
        if sim >= SIMILARITY_THRESHOLD:
            scored.append({**item, "similarity": sim})
    return sorted(scored, key=lambda x: x["similarity"], reverse=True)


def check_relation(old_text: str, new_text: str) -> dict:
    """Returns {"relation": str, "reasoning": str}."""
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{
                "role": "user",
                "content": CONTRADICTION_PROMPT.format(old_text=old_text, new_text=new_text),
            }],
            temperature=0,
            max_tokens=1024,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        relation = parsed.get("relation", "unrelated")
        reasoning = parsed.get("reasoning", "")
        if relation not in {"contradicts", "updates", "consistent", "unrelated"}:
            relation = "unrelated"
        return {"relation": relation, "reasoning": reasoning}
    except Exception as e:
        logger.error("judge failed, defaulting to 'unrelated': %s", e)
        return {"relation": "unrelated", "reasoning": "", "error": str(e)}

def resolve(new_text: str, existing_items: list[dict]) -> dict:
    """Full pipeline: find similar memories and ask the judge about the best few.
    Returns {"action": "add" | "replace" | "skip", "target_id": id | None, "reasoning": str}.
    "skip" means the statement is a near-exact repeat of an existing memory.
    If the LLM judge failed for any candidate, the result also has "llm_error".

    The top few candidates are judged (most similar first), not just the single
    best one: the memory a new statement actually updates is often NOT the most
    similar one (e.g. "I just moved to Mumbai" can look more like "I booked a
    flight to Goa" than like "I live in Delhi and work as a data analyst").
    """
    candidates = find_candidates(new_text, existing_items)
    if not candidates:
        return {"action": "add", "target_id": None, "reasoning": "no similar memory found"}

    best = candidates[0]

    # Near-identical restatement of something already stored: don't spend an
    # LLM call, and don't store a duplicate.
    if best["similarity"] >= DUPLICATE_THRESHOLD:
        return {
            "action": "skip",
            "target_id": best["id"],
            "reasoning": f"near-duplicate of existing memory (similarity {best['similarity']:.2f})",
        }

    last_reasoning = "unrelated to existing memory"
    judge_error = None
    for candidate in candidates[:MAX_CANDIDATES_TO_JUDGE]:
        result = check_relation(candidate["text"], new_text)
        if result.get("error"):
            judge_error = result["error"]
        if result["relation"] in ("contradicts", "updates"):
            return {"action": "replace", "target_id": candidate["id"], "reasoning": result["reasoning"]}
        if result["reasoning"]:
            last_reasoning = result["reasoning"]

    outcome = {"action": "add", "target_id": None, "reasoning": last_reasoning}
    if judge_error:
        outcome["llm_error"] = judge_error
    return outcome