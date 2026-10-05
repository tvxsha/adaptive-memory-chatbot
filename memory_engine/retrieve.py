"""Retrieves the memory items most relevant to a new query, so only relevant
context gets injected instead of the full history.
"""
from memory_engine.embeddings import embed, cosine_similarity

# Items less similar than this to the query are dropped, even if there are
# fewer than top_k results. Without it, unrelated memories pad the output
# (smoke test: "where does the user live?" returned a laptop decision at
# similarity 0.087, while the real answer scored 0.216).
# Tune with Vrishti's eval data.
MIN_SIMILARITY = 0.15


def retrieve_relevant(
    query: str,
    items: list[dict],
    top_k: int = 5,
    min_similarity: float = MIN_SIMILARITY,
) -> list[dict]:
    """items: from MemoryStore.get_all(). Returns up to top_k items ranked by a
    blend of semantic similarity to the query and stored importance score.
    Items below min_similarity are dropped first, so an unrelated memory can't
    get in on importance score alone.
    """
    if not items:
        return []

    query_emb = embed(query)
    ranked = []
    for item in items:
        similarity = cosine_similarity(query_emb, item["embedding"])
        if similarity < min_similarity:
            continue
        # Weighted blend — tune these weights once you have eval data.
        combined = 0.7 * similarity + 0.3 * item["score"]
        ranked.append({**item, "similarity": similarity, "combined": combined})

    ranked.sort(key=lambda x: x["combined"], reverse=True)
    return ranked[:top_k]