"""Categorizes a piece of conversation text into one of the memory buckets.

Uses Groq's Llama model with a constrained prompt. Falls back to
'recent_chat' if the model output can't be parsed, so the pipeline never
breaks on a bad LLM response.

TODO: consider few-shot examples in the prompt once you see real
misclassifications during evaluation — this is the first thing worth tuning.
"""
import json

from groq import Groq

from memory_engine.config import GROQ_API_KEY, GROQ_MODEL, CATEGORIES

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


CATEGORIZE_PROMPT = """You are classifying a single message from a conversation \
into exactly one memory category. Categories:

- fact: objective information about the user or their world (e.g. "I live in Delhi")
- preference: likes, dislikes, or standing preferences (e.g. "I hate cilantro")
- goal: something the user wants to achieve (e.g. "I'm trying to learn Spanish")
- decision: a choice the user has made (e.g. "I'm going with the blue one")
- completed_task: something finished or resolved (e.g. "I already booked the flight")
- recent_chat: small talk, filler, or anything not worth long-term storage

Respond with ONLY a JSON object of the form:
{{"category": "<one of the categories above>", "confidence": <0.0-1.0>}}

Message: "{text}"
"""


def categorize(text: str) -> dict:
    """Returns {"category": str, "confidence": float}."""
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": CATEGORIZE_PROMPT.format(text=text)}],
            temperature=0,
            max_tokens=1024,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if the model adds them anyway
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        category = parsed.get("category", "recent_chat")
        confidence = float(parsed.get("confidence", 0.5))
        if category not in CATEGORIES:
            category = "recent_chat"
        return {"category": category, "confidence": confidence}
    except Exception as e:
        print(f"[categorize] fell back to 'recent_chat' due to: {e}")
        return {"category": "recent_chat", "confidence": 0.0}
