"""
app/training/entities.py — Per-entity binary classifier training.

For each entity of type "train", trains a LogisticRegression that detects
whether the entity is present in a sentence (binary: 0/1).

Negative samples are sourced in priority order:
  1. Other trainable entities' examples (real confusions)
  2. Workflow context sentences (intent examples)
  3. Generic fallback phrases (last resort)
"""

import logging
import numpy as np
from typing import Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import app.state as state

logger = logging.getLogger("mybot-nlu")

_GENERIC_FALLBACK = [
    "hello", "hi there", "yes", "no", "okay", "sure", "thanks",
    "what is this", "how are you", "help me", "i need", "can you",
    "please help", "thank you", "goodbye", "see you", "have a good day",
]


def train_entity_classifiers(
    workflow_id: str,
    entities: Optional[list] = None,
    context_sentences: Optional[list[str]] = None,
) -> dict:
    """
    Train per-entity binary classifiers for a workflow.

    Returns:
        { entity_name: { totalExamples, totalSamples, cvAccuracy, ... } }
    """
    if not state.intent_model:
        raise RuntimeError("Sentence encoder is not loaded yet.")

    if not entities:
        logger.info(f"No entity definitions for '{workflow_id}' — skipping entity training")
        return {}

    entity_metrics:               dict = {}
    entity_classifiers_for_wf:   dict = {}
    context_sentences = list(context_sentences or [])

    # Collect all trainable entities upfront so each entity can use
    # the others' examples as hard negatives.
    trainable_entities: list[tuple[str, list[str]]] = []
    for ent in entities:
        e_type     = ent.get("type", "regex")    if isinstance(ent, dict) else getattr(ent, "type", "regex")
        e_name     = ent.get("name")             if isinstance(ent, dict) else getattr(ent, "name", None)
        e_examples = ent.get("examples", [])     if isinstance(ent, dict) else getattr(ent, "examples", [])
        if e_type == "train" and e_name and e_examples:
            trainable_entities.append((e_name, list(e_examples)))

    for ent in entities:
        ent_type = ent.get("type", "regex")   if isinstance(ent, dict) else getattr(ent, "type", "regex")
        ent_name = ent.get("name")            if isinstance(ent, dict) else getattr(ent, "name", None)
        examples = ent.get("examples", [])    if isinstance(ent, dict) else getattr(ent, "examples", [])

        if ent_type != "train" or not ent_name or len(examples) < 2:
            if ent_name and ent_type == "train" and len(examples) < 2:
                logger.warning(f"  ⚠️  Entity '{ent_name}' needs at least 2 examples — skipping")
            continue

        logger.info(f"  Training entity classifier for '{ent_name}' ({len(examples)} examples)")

        positive_samples      = list(examples)
        target_negative_count = len(positive_samples) * 2
        negative_samples:     list[str] = []
        seen_negatives:       set[str]  = set()

        def _add_neg(phrase: str):
            key = phrase.strip().lower()
            if key and key not in seen_negatives and len(negative_samples) < target_negative_count:
                seen_negatives.add(key)
                negative_samples.append(phrase.strip())

        for other_name, other_examples in trainable_entities:
            if other_name != ent_name:
                for ex in other_examples:
                    _add_neg(ex)

        for sent in context_sentences:
            _add_neg(sent)

        fb_idx = 0
        while len(negative_samples) < target_negative_count and fb_idx < len(_GENERIC_FALLBACK) * 3:
            _add_neg(_GENERIC_FALLBACK[fb_idx % len(_GENERIC_FALLBACK)])
            fb_idx += 1
            if fb_idx >= len(_GENERIC_FALLBACK) and len(negative_samples) < target_negative_count:
                break

        negative_source = (
            "cross-entity+context"
            if (len(trainable_entities) > 1 or context_sentences)
            else "generic-fallback-only"
        )

        try:
            X_pos = state.intent_model.encode(positive_samples, show_progress_bar=False, batch_size=32)
            X_neg = state.intent_model.encode(negative_samples, show_progress_bar=False, batch_size=32)
            X     = np.vstack([X_pos, X_neg])
            y     = np.array([1] * len(X_pos) + [0] * len(X_neg))

            clf = LogisticRegression(
                C=1.0, max_iter=500, class_weight="balanced",
                solver="lbfgs", random_state=42,
            )

            cv_accuracy     = None
            min_class_count = min(len(X_pos), len(X_neg))
            if len(y) >= 4 and min_class_count >= 2:
                try:
                    cv_scores   = cross_val_score(clf, X, y, cv=min(3, min_class_count), scoring="accuracy")  # type: ignore
                    cv_accuracy = round(float(cv_scores.mean()), 4)
                    logger.info(f"    CV accuracy for entity '{ent_name}': {cv_accuracy:.2%}")
                except Exception as e:
                    logger.warning(f"    CV skipped for entity '{ent_name}': {e}")

            clf.fit(X, y)  # type: ignore
            entity_classifiers_for_wf[ent_name] = clf
            entity_metrics[ent_name] = {
                "totalExamples":  len(examples),
                "totalSamples":   len(y),
                "cvAccuracy":     cv_accuracy,
                "positives":      len(X_pos),
                "negatives":      len(X_neg),
                "negativeSource": negative_source,
            }
            logger.info(f"  ✓ Entity classifier trained for '{ent_name}' (negatives: {negative_source})")

        except Exception as e:
            logger.error(f"  ✗ Failed to train entity classifier for '{ent_name}': {e}")

    if entity_classifiers_for_wf:
        state.trained_entity_classifiers[workflow_id] = entity_classifiers_for_wf

    return entity_metrics
