"""
app/inference/intent.py — Intent / subintent detection at runtime.

Two modes:
  - Trained:   LogisticRegression classifier (fast, accurate)
  - Zero-shot: Cosine similarity fallback when no trained model exists
"""

import logging
from typing import Optional

from sentence_transformers import util

import app.state as state
from app.models import SubIntent

logger = logging.getLogger("mybot-nlu")


def detect_intent_trained(text: str, workflow_id: str, model_data: dict, threshold: float = 0.30) -> dict:
    """Use the trained LogisticRegression classifier for a workflow."""
    if state.intent_model is None:
        raise RuntimeError("intent_model is not loaded yet")

    clf = model_data["classifier"]
    le  = model_data["label_encoder"]

    embedding     = state.intent_model.encode([text])
    probabilities = clf.predict_proba(embedding)[0]
    intent_names  = le.classes_

    all_scores = [
        {"name": name, "label": name, "score": round(float(prob), 4)}
        for name, prob in zip(intent_names, probabilities)
    ]
    all_scores.sort(key=lambda x: x["score"], reverse=True)
    best = all_scores[0]

    logger.info(f"[TRAINED:{workflow_id}] Intent: {best['name']} (prob={best['score']})")

    if best["score"] < threshold:
        return {
            "intent": "fallback", "subintent": "fallback",
            "confidence": best["score"], "allScores": all_scores, "mode": "trained",
        }
    return {
        "intent": best["name"], "subintent": best["name"],
        "confidence": best["score"], "allScores": all_scores, "mode": "trained",
    }


def detect_intent_zeroshot(
    text: str,
    subIntents: list[SubIntent],
    threshold: float = 0.30,
) -> dict:
    """Cosine-similarity zero-shot fallback when no trained model exists."""
    if state.intent_model is None:
        raise RuntimeError("intent_model is not loaded yet")

    if not subIntents:
        return {"intent": "unknown", "subintent": "unknown", "confidence": 0.0, "allScores": [], "mode": "zeroshot"}

    user_embedding = state.intent_model.encode(text, convert_to_tensor=True)
    scores = []

    for sub in subIntents:
        refs = list(sub.examples or [])
        if sub.label:       refs.append(sub.label)
        if sub.description: refs.append(sub.description)

        if not refs:
            scores.append({"name": sub.name, "label": sub.label, "score": 0.0})
            continue

        ref_embeddings = state.intent_model.encode(refs, convert_to_tensor=True)
        similarities   = util.cos_sim(user_embedding, ref_embeddings)[0]
        best_score     = float(similarities.max().item())
        scores.append({"name": sub.name, "label": sub.label, "score": round(best_score, 4)})

    scores.sort(key=lambda x: x["score"], reverse=True)
    best = scores[0]
    logger.info(f"[ZEROSHOT] Intent: {best['name']} (score={best['score']})")

    if best["score"] < threshold:
        return {"intent": "fallback", "subintent": "fallback", "confidence": best["score"], "allScores": scores, "mode": "zeroshot"}
    return {"intent": best["name"], "subintent": best["name"], "confidence": best["score"], "allScores": scores, "mode": "zeroshot"}


def detect_intent_global(text: str, workflow_id: str, subIntents: list[SubIntent], threshold: float = 0.30) -> dict:
    """Use the global shared classifiers to predict intent and mask by workflow."""
    if state.intent_model is None:
        raise RuntimeError("intent_model is not loaded yet")

    clf = state.shared_intent_classifier
    le = state.shared_intent_label_encoder

    embedding = state.intent_model.encode([text])
    
    # Optional: We could use state.shared_workflow_classifier here to verify or predict workflow
    # But since workflow_id is usually passed in, we use the workflow_metadata masking approach.
    
    probabilities = clf.predict_proba(embedding)[0]
    intent_names = le.classes_

    valid_intent_names = {sub.name for sub in subIntents if sub.name}

    all_scores = []
    for name, prob in zip(intent_names, probabilities):
        if name in valid_intent_names:
            all_scores.append({"name": name, "label": name, "score": round(float(prob), 4)})
            
    if not all_scores:
        return detect_intent_zeroshot(text, subIntents, threshold)

    all_scores.sort(key=lambda x: x["score"], reverse=True)
    best = all_scores[0]

    logger.info(f"[GLOBAL_TRAINED:{workflow_id}] Intent: {best['name']} (prob={best['score']})")

    if best["score"] < threshold:
        return {
            "intent": "fallback", "subintent": "fallback",
            "confidence": best["score"], "allScores": all_scores, "mode": "global_trained",
        }
    return {
        "intent": best["name"], "subintent": best["name"],
        "confidence": best["score"], "allScores": all_scores, "mode": "global_trained",
    }


def detect_intent(
    text: str,
    workflow_id: str,
    subIntents: list[SubIntent],
    threshold: float = 0.30,
) -> dict:
    """Auto-selects global, trained, or zero-shot based on availability."""
    if state.shared_intent_classifier is not None:
        return detect_intent_global(text, workflow_id, subIntents, threshold)

    from app.core.model_registry import registry
    model_data = registry.get_model(workflow_id)
    if model_data:
        return detect_intent_trained(text, workflow_id, model_data, threshold)
    return detect_intent_zeroshot(text, subIntents, threshold)
