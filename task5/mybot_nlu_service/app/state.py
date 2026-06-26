"""
app/state.py — Global runtime state shared across all modules.

All mutable globals live here so every module imports from one place.
Never mutate these dicts outside this module's helper functions.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ─── Base model handles (set during lifespan startup) ─────────────────────────
spacy_nlp        = None   # spaCy en_core_web_sm
intent_model     = None   # SentenceTransformer all-MiniLM-L6-v2
models_loaded    = False
model_load_error = None

# ─── Per-workflow intent classifiers ──────────────────────────────────────────
# Keyed by workflowId string.
trained_classifiers:    dict[str, LogisticRegression] = {}
trained_label_encoders: dict[str, LabelEncoder]       = {}
trained_metadata_store: dict[str, dict]               = {}

# ─── Per-workflow per-entity binary classifiers ───────────────────────────────
# { workflowId: { entityName: LogisticRegression } }
trained_entity_classifiers: dict[str, dict[str, LogisticRegression]] = {}
