"""
MyBot NLU Service  (v3.2 — Per-Workflow Multi-Model with spaCy)
==============================================================

Entry point — app creation, lifespan (model loading), and router registration.
All business logic lives in the app/ package.

Endpoints:
  POST /train             → Full retrain (re-encodes everything)
  POST /train/incremental → Incremental retrain (skips unchanged intents)
  POST /analyze           → Full NLU pipeline (intent + entities)
  POST /detect-intent     → Intent detection only
  POST /extract-entities  → Entity extraction only
  GET  /health            → Service health + loaded models
  GET  /training-status   → Per-workflow training metadata
  DELETE /training-status → Delete a trained model (revert to zero-shot)
  GET  /                  → Root check
"""

import logging
import os
import subprocess
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

import app.state as state
from app.config import TRAINED_MODELS_DIR
from app.core.persistence import load_trained_classifier
from app.routers import train as train_router
from app.routers import analyze as analyze_router
from app.routers import status as status_router

logger = logging.getLogger("mybot-nlu")


# ─── Lifespan: model loading + auto-restore persisted classifiers ─────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load base models on startup; auto-load all previously trained classifiers."""
    global_state = state

    try:
        logger.info("Loading spaCy model: en_core_web_sm ...")
        try:
            import spacy
            global_state.spacy_nlp = spacy.load("en_core_web_sm")
            logger.info("✓ spaCy model loaded")
        except OSError:
            logger.warning("✗ spaCy model not found — attempting auto-download ...")
            try:
                import spacy
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                global_state.spacy_nlp = spacy.load("en_core_web_sm")
                logger.info("✓ spaCy model downloaded and loaded")
            except Exception as e:
                logger.error(f"✗ spaCy download failed: {e}. Continuing without spaCy NER.")

        logger.info("Loading encoder: sentence-transformers/all-MiniLM-L6-v2 ...")
        global_state.intent_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("✓ Sentence encoder loaded")

        global_state.models_loaded = True
        logger.info("✓ Base models ready")

    except Exception as e:
        global_state.model_load_error = str(e)
        logger.error(f"✗ Model loading failed: {e}")
        yield
        return

    # Auto-restore all previously trained classifiers from disk
    loaded_count = 0
    for pkl_file in TRAINED_MODELS_DIR.glob("*.pkl"):
        workflow_id = pkl_file.stem
        try:
            load_trained_classifier(workflow_id)
            logger.info(f"  ✓ Auto-loaded classifier: {workflow_id}")
            loaded_count += 1
        except Exception as e:
            logger.warning(f"  ✗ Could not load classifier '{workflow_id}': {e}")

    logger.info(f"✓ NLU service fully live — {loaded_count} trained model(s) loaded")
    yield
    # Shutdown: nothing to clean up


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="MyBot NLU Service",
    description=(
        "Per-workflow trainable intent classifier (LogisticRegression + SentenceTransformers) "
        "+ spaCy NER entity extraction. v3.2 — Incremental training + 80/20 train/test split."
    ),
    version="3.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3003",
        "http://localhost:3000",
        os.getenv("FRONTEND_URL", "http://localhost:3003")
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# ─── Register routers ─────────────────────────────────────────────────────────

app.include_router(train_router.router)
app.include_router(analyze_router.router)
app.include_router(status_router.router)


# ─── Root ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message":          "MyBot NLU Service v3.2 is running",
        "trainedWorkflows": list(state.trained_classifiers.keys()),
        "docs":             "/docs",
    }