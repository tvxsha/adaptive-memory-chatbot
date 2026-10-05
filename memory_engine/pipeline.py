"""Wires categorize -> score -> contradiction-check -> store into the single
'process a new message' step described in the methodology slide.

Every step logs what it did, so when a conversation behaves oddly you can
read the server terminal and see exactly where the decision was made.
"""
import logging

from memory_engine.categorize import categorize
from memory_engine.score import score_importance
from memory_engine.contradiction import resolve
from memory_engine.embeddings import embed
from memory_engine.store import MemoryStore

logger = logging.getLogger("memory_engine.pipeline")

# Messages longer than this are cut before being sent to the LLM and the
# embedding API. Keeps token usage and latency predictable on pasted walls
# of text (code blocks, long documents).
MAX_MESSAGE_CHARS = 2000

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

    Possible "action" values: "skip", "added", "replaced".
    """
    # --- Step 0: input guards -------------------------------------------
    if text is None or not text.strip():
        logger.info("[%s] skipped: empty message", conversation_id)
        return {"action": "skip", "category": None, "reason": "empty message"}

    text = text.strip()
    if len(text) > MAX_MESSAGE_CHARS:
        logger.info(
            "[%s] truncating message from %d to %d chars",
            conversation_id, len(text), MAX_MESSAGE_CHARS,
        )
        text = text[:MAX_MESSAGE_CHARS]

    store = get_store()

    # --- Step 1: categorize ---------------------------------------------
    category_result = categorize(text)
    category = category_result["category"]
    if category_result.get("error"):
        logger.error("[%s] categorizer failed, message NOT processed", conversation_id)
        return {
            "action": "error",
            "category": None,
            "reason": f"categorize failed: {category_result['error']}",
        }
    logger.info(
        "[%s] categorized as %s (confidence %.2f): %r",
        conversation_id, category, category_result["confidence"], text[:80],
    )

    # recent_chat items aren't worth long-term storage or contradiction
    # checking — skip straight through.
    if category == "recent_chat":
        logger.info("[%s] skipped: recent_chat filler", conversation_id)
        return {"action": "skip", "category": category, "reason": "recent_chat filler"}

    # --- Step 2: score + embed ------------------------------------------
    score = score_importance(text, category)
    embedding = embed(text)
    logger.info("[%s] importance score = %.3f", conversation_id, score)

    # --- Step 3: contradiction / duplicate check ------------------------
    existing_items = store.get_all(conversation_id)
    decision = resolve(text, existing_items)
    logger.info(
        "[%s] contradiction check -> %s (%s)",
        conversation_id, decision["action"], decision["reasoning"],
    )

    # --- Step 4: update the store ---------------------------------------
    if decision["action"] == "skip":
        return {
            "action": "skip",
            "category": category,
            "reason": decision["reasoning"],
        }

    if decision["action"] == "replace":
        store.replace(decision["target_id"], text, category, score, embedding)
        action = "replaced"
    else:
        store.add(conversation_id, text, category, score, embedding)
        action = "added"

    logger.info("[%s] memory %s", conversation_id, action)
    result = {
        "action": action,
        "category": category,
        "score": score,
        "reasoning": decision["reasoning"],
    }
    if decision.get("llm_error"):
        logger.warning("[%s] stored without a working contradiction check", conversation_id)
        result["llm_error"] = decision["llm_error"]
    return result