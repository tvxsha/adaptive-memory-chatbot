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

from groq import Groq

from memory_engine.config import GROQ_API_KEY, GROQ_MODEL
from memory_engine.embeddings import embed, cosine_similarity

SIMILARITY_THRESHOLD = 0.55

_client = None


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

- "contradicts": the new statement directly conflicts with the old one and both can't be true
- "updates": the new statement is a natural progression/update of the old one (e.g. moved cities)
- "consistent": both can be true together, no conflict
- "unrelated": they aren't about the same topic
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
            max_tokens=150,
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
        print(f"[contradiction] defaulted to 'unrelated' due to: {e}")
        return {"relation": "unrelated", "reasoning": ""}


def resolve(new_text: str, existing_items: list[dict]) -> dict:
    """Full pipeline: find the best candidate and classify the relation.
    Returns {"action": "add" | "replace" | "skip", "target_id": id | None, "reasoning": str}.
    """
    candidates = find_candidates(new_text, existing_items)
    if not candidates:
        return {"action": "add", "target_id": None, "reasoning": "no similar memory found"}

    best = candidates[0]
    result = check_relation(best["text"], new_text)

    if result["relation"] in ("contradicts", "updates"):
        return {"action": "replace", "target_id": best["id"], "reasoning": result["reasoning"]}
    if result["relation"] == "consistent":
        return {"action": "add", "target_id": None, "reasoning": result["reasoning"]}
    return {"action": "add", "target_id": None, "reasoning": "unrelated to existing memory"}
