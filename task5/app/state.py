"""
app/state.py — Global runtime state shared across all modules.

All mutable globals live here so every module imports from one place.
Never mutate these dicts outside this module's helper functions.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from typing import Any

# ─── Base model handles (set during lifespan startup) ─────────────────────────
spacy_nlp: Any = None   # spaCy en_core_web_sm
intent_model: Any = None   # SentenceTransformer all-MiniLM-L6-v2
models_loaded: bool = False
model_load_error: Any = None

# ─── Per-workflow intent classifiers ──────────────────────────────────────────
# Keyed by workflowId string.
trained_classifiers:    dict[str, LogisticRegression] = {}
trained_label_encoders: dict[str, LabelEncoder]       = {}
trained_metadata_store: dict[str, dict]               = {}

# ─── Per-workflow per-entity binary classifiers ───────────────────────────────
# { workflowId: { entityName: LogisticRegression } }
trained_entity_classifiers: dict[str, dict[str, LogisticRegression]] = {}

# ─── Global Shared classifiers (Task 2) ───────────────────────────────────────
shared_intent_classifier:      Any = None
shared_intent_label_encoder:   Any = None
shared_workflow_classifier:    Any = None
shared_workflow_label_encoder: Any = None
