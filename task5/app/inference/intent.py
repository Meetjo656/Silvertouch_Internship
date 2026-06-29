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


def detect_intent_trained(text: str, workflow_id: str, threshold: float = 0.30) -> dict:
    """Use the trained LogisticRegression classifier for a workflow."""
    clf = state.trained_classifiers[workflow_id]
    le  = state.trained_label_encoders[workflow_id]

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
    if not subIntents:
        return {"intent": "unknown", "subintent": "unknown", "confidence": 0.0, "allScores": [], "mode": "zeroshot"}

    user_embedding = state.intent_model.encode(text, convert_to_tensor=True)
    scores = []

    for sub in subIntents:
        refs = list(sub.examples)
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


def detect_intent(
    text: str,
    workflow_id: str,
    subIntents: list[SubIntent],
    threshold: float = 0.30,
) -> dict:
    """Auto-selects trained or zero-shot based on whether a model exists."""
    if workflow_id in state.trained_classifiers:
        return detect_intent_trained(text, workflow_id, threshold)
    return detect_intent_zeroshot(text, subIntents, threshold)
