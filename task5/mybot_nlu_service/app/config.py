"""
app/config.py — Centralised configuration for the NLU service.

All directory paths are driven by environment variables so they can be
overridden in production without touching code.
"""

import os
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("mybot-nlu")

# ─── Storage directories ──────────────────────────────────────────────────────

TRAINED_MODELS_DIR = Path(os.getenv("TRAINED_MODELS_DIR", "./trained_models"))
TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_DATA_DIR = Path(os.getenv("TRAINING_DATA_DIR", "./training_data"))
TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Pre-computed intent embeddings — enables incremental retraining.
EMBEDDING_CACHE_DIR = Path(os.getenv("EMBEDDING_CACHE_DIR", "./embedding_cache"))
EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ─── API Authentication ────────────────────────────────────────────────────────────
NLU_API_KEY = os.getenv("NLU_API_KEY", "")
