"""Embedding wrapper using the Hugging Face Inference API (hosted), instead
of running sentence-transformers/PyTorch locally. This keeps the install
lightweight — no compiled PyTorch dependency — at the cost of needing an
internet connection and a free HF token for every embedding call.

If you'd rather run embeddings fully offline/locally (e.g. once everyone's
on Python 3.12 and torch installs cleanly), swap this file for a version
using `sentence-transformers.SentenceTransformer` instead — the public
functions below (embed, embed_batch, cosine_similarity) can keep the same
signatures so nothing else in memory_engine needs to change.
"""
import numpy as np
from huggingface_hub import InferenceClient

from memory_engine.config import EMBEDDING_MODEL, HF_TOKEN

_client = None


def _get_client() -> InferenceClient:
    global _client
    if _client is None:
        _client = InferenceClient(model=EMBEDDING_MODEL, token=HF_TOKEN)
    return _client


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def embed(text: str) -> np.ndarray:
    """Return a single normalized embedding vector for a piece of text."""
    client = _get_client()
    result = client.feature_extraction(text, normalize=True)
    vec = np.array(result, dtype=np.float32)
    # Some models return per-token embeddings (2D) instead of a single
    # pooled vector — mean-pool across tokens if so.
    if vec.ndim == 2:
        vec = vec.mean(axis=0)
    return _normalize(vec)


def embed_batch(texts: list[str]) -> np.ndarray:
    """Returns a 2D array, one normalized embedding row per input text."""
    return np.stack([embed(t) for t in texts])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Vectors are already normalized by embed(), so this is just a dot product."""
    return float(np.dot(a, b))
