"""Scores a memory item by how useful it's likely to be later — not just how
recent it is. This starts as a simple heuristic combining category weight and
text specificity; swap in a learned or LLM-scored version once you have
evaluation data to tune against.

TODO: this is the most impactful piece to iterate on for your evaluation
section — try comparing this heuristic scorer against an LLM-based scorer
and report the difference.
"""
import math

# Base importance weight per category — facts/decisions/goals tend to matter
# longer than recent_chat filler. Tune these based on your eval results.
CATEGORY_WEIGHTS = {
    "fact": 0.9,
    "preference": 0.8,
    "goal": 0.85,
    "decision": 0.85,
    "completed_task": 0.5,
    "recent_chat": 0.2,
}


def specificity_bonus(text: str) -> float:
    """Longer, more detailed statements tend to carry more retrievable
    information than short filler. Capped so it doesn't dominate the score.
    """
    word_count = len(text.split())
    return min(word_count / 30.0, 0.2)


def score_importance(text: str, category: str) -> float:
    """Returns a score in roughly [0, 1.1], higher = more worth keeping."""
    base = CATEGORY_WEIGHTS.get(category, 0.3)
    bonus = specificity_bonus(text)
    return round(base + bonus, 3)


def recency_decay(age_in_turns: int, half_life: int = 20) -> float:
    """Optional multiplier if you want recency to matter *some* — decays
    slowly so it never overrides a genuinely important old fact on its own.
    Not applied by default in store.py; wire it in if your evaluation shows
    pure importance-scoring loses too much temporal signal.
    """
    return math.exp(-age_in_turns / half_life)
