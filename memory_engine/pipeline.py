"""Wires categorize -> score -> contradiction-check -> store into the single
'process a new message' step described in the methodology slide.
"""
from memory_engine.categorize import categorize
from memory_engine.score import score_importance
from memory_engine.contradiction import resolve
from memory_engine.embeddings import embed
from memory_engine.store import MemoryStore

_store = None


def get_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store


def process_message(conversation_id: str, text: str) -> dict:
    """Runs one message through the full pipeline. Returns a summary of what
    happened, useful for the extension's live memory panel and for logging
    during evaluation.
    """
    store = get_store()

    category_result = categorize(text)
    category = category_result["category"]

    # recent_chat items aren't worth long-term storage or contradiction
    # checking — skip straight through.
    if category == "recent_chat":
        return {"action": "skip", "category": category, "reason": "recent_chat filler"}

    score = score_importance(text, category)
    embedding = embed(text)

    existing_items = store.get_all(conversation_id)
    decision = resolve(text, existing_items)

    if decision["action"] == "replace":
        store.replace(decision["target_id"], text, category, score, embedding)
        action = "replaced"
    else:
        store.add(conversation_id, text, category, score, embedding)
        action = "added"

    return {
        "action": action,
        "category": category,
        "score": score,
        "reasoning": decision["reasoning"],
    }
