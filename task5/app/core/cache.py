"""
app/core/cache.py — Embedding cache helpers (per-workflow .npz files).

Stores pre-computed sentence-transformer embeddings per intent so incremental
retraining can skip re-encoding unchanged subintents.
"""

import re
import logging
import numpy as np
from pathlib import Path

from app.config import EMBEDDING_CACHE_DIR

logger = logging.getLogger("mybot-nlu")


def safe_np_key(name: str) -> str:
    """Sanitize an intent name into a valid numpy .npz key (no special chars)."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def embedding_cache_path(workflow_id: str) -> Path:
    return EMBEDDING_CACHE_DIR / f"{workflow_id}.npz"


def load_embedding_cache(workflow_id: str) -> dict[str, np.ndarray]:
    """
    Load cached embeddings for a workflow.
    Returns { sanitized_intent_name: np.ndarray }, or {} if no cache exists.
    """
    path = embedding_cache_path(workflow_id)
    if not path.exists():
        return {}
    try:
        data = np.load(str(path), allow_pickle=False)
        return {k: data[k] for k in data.files}
    except Exception as e:
        logger.warning(f"Could not load embedding cache for '{workflow_id}': {e}")
        return {}


def save_embedding_cache(workflow_id: str, embeddings_per_intent: dict[str, np.ndarray]) -> None:
    """Persist embeddings for every intent in this workflow."""
    path = embedding_cache_path(workflow_id)
    try:
        safe_kv = {safe_np_key(k): v for k, v in embeddings_per_intent.items()}
        np.savez(str(path), **safe_kv)
        logger.info(f"Embedding cache saved for '{workflow_id}' ({len(embeddings_per_intent)} intents)")
    except Exception as e:
        logger.warning(f"Could not save embedding cache for '{workflow_id}': {e}")


def clear_embedding_cache(workflow_id: str) -> None:
    """Delete the embedding cache for a workflow (used before a full retrain)."""
    path = embedding_cache_path(workflow_id)
    if path.exists():
        path.unlink()
        logger.info(f"Embedding cache cleared for '{workflow_id}'")
