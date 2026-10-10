"""Embedding wrapper.

Default: run all-MiniLM-L6-v2 LOCALLY with fastembed (ONNX, no PyTorch needed,
free, no rate limits; the ~90 MB model is downloaded once on first use).
Fallback: the Hugging Face Inference API, used only if fastembed is not
installed. The public functions (embed, embed_batch, cosine_similarity) are the
same either way, so nothing else in memory_engine needs to change.
"""
import numpy as np

from memory_engine.config import EMBEDDING_MODEL, HF_TOKEN

try:
    from fastembed import TextEmbedding
except ImportError:  # fall back to the hosted API
    TextEmbedding = None

_local_model = None
_client = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        _local_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _local_model


def _get_client():
    global _client
    if _client is None:
        from huggingface_hub import InferenceClient

        _client = InferenceClient(model=EMBEDDING_MODEL, token=HF_TOKEN)
    return _client


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def embed(text: str) -> np.ndarray:
    """Return a single normalized embedding vector for a piece of text."""
    if TextEmbedding is not None:
        vec = np.array(next(iter(_get_local_model().embed([text]))), dtype=np.float32)
    else:
        result = _get_client().feature_extraction(text, normalize=True)
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