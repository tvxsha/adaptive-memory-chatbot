"""Retrieves the memory items most relevant to a new query, so only relevant
context gets injected instead of the full history.
"""
from memory_engine.embeddings import embed, cosine_similarity


def retrieve_relevant(query: str, items: list[dict], top_k: int = 5) -> list[dict]:
    """items: from MemoryStore.get_all(). Returns top_k items ranked by a
    blend of semantic similarity to the query and stored importance score,
    so a highly-relevant-but-low-importance item can still surface, and a
    very-important item close to the topic isn't buried.
    """
    if not items:
        return []

    query_emb = embed(query)
    ranked = []
    for item in items:
        similarity = cosine_similarity(query_emb, item["embedding"])
        # Weighted blend — tune these weights once you have eval data.
        combined = 0.7 * similarity + 0.3 * item["score"]
        ranked.append({**item, "similarity": similarity, "combined": combined})

    ranked.sort(key=lambda x: x["combined"], reverse=True)
    return ranked[:top_k]
