"""
app/training/intent.py — Full and incremental intent classifier training.

Both paths share _fit_classifier_and_evaluate() which:
  1. Runs the 80/20 held-out accuracy evaluation.
  2. Fits the production LogisticRegression on ALL data.
  3. Persists the model + metadata.
"""

import time
import logging
import numpy as np
from typing import Optional

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

import app.state as state
from app.models import SubIntent
from app.core.augmentation  import augment_examples
from app.core.cache         import load_embedding_cache, save_embedding_cache, clear_embedding_cache, safe_np_key
from app.core.persistence   import save_trained_classifier, save_training_data, load_training_data, model_path
from app.core.fingerprint   import compute_intent_fingerprints
from app.core.evaluation    import evaluate_train_test_split
from app.training.entities  import train_entity_classifiers
from app.core.persistence import (
    save_current_as_version,
    keep_only_last_n_versions,
    list_versions,
    get_next_version_number,
    version_path,
    training_data_path,
    TRAINED_MODELS_DIR,
    TRAINING_DATA_DIR,
    load_trained_classifier
)

logger = logging.getLogger("mybot-nlu")


# ─── Entity detail helpers ────────────────────────────────────────────────────

def _build_entity_details_map(workflow_id: str, entities: Optional[list]) -> dict:
    """Build name → {type, regex, examples} from disk + current request payload."""
    entity_details_map: dict = {}
    saved = load_training_data(workflow_id)
    if saved:
        for ent in saved.get("entities", []):
            if ent.get("name"):
                entity_details_map[ent["name"]] = {
                    "type":     ent.get("type", "regex"),
                    "regex":    ent.get("regex", ""),
                    "examples": ent.get("examples", []),
                }
    for ent in (entities or []):
        ent_name = getattr(ent, "name", None) or (ent.get("name") if isinstance(ent, dict) else None)
        if ent_name:
            entity_details_map[ent_name] = {
                "type":     getattr(ent, "type", "regex")  or (ent.get("type", "regex")  if isinstance(ent, dict) else "regex"),
                "regex":    getattr(ent, "regex", "")      or (ent.get("regex", "")      if isinstance(ent, dict) else ""),
                "examples": getattr(ent, "examples", [])   or (ent.get("examples", [])   if isinstance(ent, dict) else []),
            }
    return entity_details_map


def _build_required_entities_map(subIntents: list, entity_details_map: dict) -> dict:
    """Build subintent-name → [entity configs] for runtime entity checks."""
    required_entities_map: dict = {}
    for sub in subIntents:
        name = getattr(sub, "name", None) or (sub.get("name") if isinstance(sub, dict) else None)
        if not name:
            continue
        req_ents = (
            getattr(sub, "requiredEntities", None)
            or (sub.get("requiredEntities") if isinstance(sub, dict) else None)
            or []
        )
        entity_configs = []
        for ent in req_ents:
            if isinstance(ent, str):
                details = entity_details_map.get(ent, {"type": "regex", "regex": "", "examples": []})
                entity_configs.append({"name": ent, **{k: details.get(k, "") for k in ("type", "regex", "examples")}})
            elif isinstance(ent, dict):
                ent_name = ent.get("name", "")
                details  = entity_details_map.get(ent_name, {})
                entity_configs.append({
                    "name":     ent_name,
                    "type":     ent.get("type")     or details.get("type", "regex"),
                    "regex":    ent.get("regex")    or details.get("regex", ""),
                    "examples": ent.get("examples") or details.get("examples", []),
                })
        required_entities_map[name] = entity_configs
    return required_entities_map


# ─── Shared fit + evaluate ────────────────────────────────────────────────────

def _fit_classifier_and_evaluate(
    workflow_id: str,
    sentences: list[str],
    labels: list[str],
    X: np.ndarray,
    augmentation_factor: int,
    entities: Optional[list],
    subIntents: list,
    raw_context_sentences: list[str],
    encode_time: float,
    incremental: bool = False,
    changed_intents: Optional[set] = None,
    cache_hits: int = 0,
    cache_misses: int = 0,
) -> dict:
    """
    Shared tail for both full and incremental training:
    1. 80/20 held-out accuracy evaluation.
    2. Fit production classifier on ALL data.
    3. Persist model + metadata + entity classifiers.
    """
    le = LabelEncoder()
    y  = le.fit_transform(labels)

    split_eval = evaluate_train_test_split(X, y, le, workflow_id)

    clf = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced",
        solver="lbfgs", random_state=42,
    )

    cv_accuracy     = None
    class_counts    = np.bincount(y)
    min_class_count = int(class_counts.min()) if len(class_counts) else 0
    if len(sentences) >= 10 and len(set(labels)) >= 2 and min_class_count >= 2:
        try:
            cv_folds  = min(3, min_class_count)
            cv_scores = cross_val_score(clf, X, y, cv=cv_folds, scoring="accuracy")
            cv_accuracy = round(float(cv_scores.mean()), 4)
            logger.info(
                f"CV accuracy for '{workflow_id}': {cv_accuracy:.2%}  |  "
                f"Test accuracy: {split_eval.get('testAccuracy')}"
            )
        except Exception as e:
            logger.warning(f"CV skipped for '{workflow_id}': {e}")

    t1 = time.time()
    clf.fit(X, y)
    train_time = round(time.time() - t1, 2)
    logger.info(f"Final classifier fitted in {train_time}s")

    state.trained_classifiers[workflow_id]    = clf
    state.trained_label_encoders[workflow_id] = le

    entity_details_map    = _build_entity_details_map(workflow_id, entities)
    required_entities_map = _build_required_entities_map(subIntents, entity_details_map)
    entity_metrics        = train_entity_classifiers(
        workflow_id, entities, context_sentences=raw_context_sentences
    )

    state.trained_metadata_store[workflow_id] = {
        "workflowId":           workflow_id,
        "trainedAt":            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "totalSentences":       len(sentences),
        "totalIntents":         len(set(labels)),
        "intents":              list(le.classes_),
        "augmentationFactor":   augmentation_factor,
        "encodingTimeS":        encode_time,
        "trainingTimeS":        train_time,
        "modelFile":            str(model_path(workflow_id)),
        "requiredEntitiesMap":  required_entities_map,
        "entityMetrics":        entity_metrics,
        "totalTrainedEntities": len(entity_metrics),
        "trainingMode":         "incremental" if incremental else "full",
        "changedIntents":       sorted(changed_intents) if changed_intents else None,
        "cacheHits":            cache_hits,
        "cacheMisses":          cache_misses,
        # Accuracy metrics
        "cvAccuracy":           cv_accuracy,
        "testAccuracy":         split_eval.get("testAccuracy"),
        "trainAccuracy":        split_eval.get("trainAccuracy"),
        "testSamples":          split_eval.get("testSamples"),
        "trainSamples":         split_eval.get("trainSamples"),
        "perClassReport":       split_eval.get("perClassReport"),
        "splitSkipped":         split_eval.get("splitSkipped"),
        "splitSkipReason":      split_eval.get("splitSkipReason"),
        "overfitWarning":       split_eval.get("overfitWarning", False),
        "overfitGap":           split_eval.get("overfitGap", 0.0),
    }

    # ─── Version Management ────────────────────────────────────────────────────
    # Get current version info before saving new model
    # Get current version info before saving new model
    old_versions = list_versions(workflow_id)
    old_cv_acc = None
    old_active_version_num = 1
    for v in old_versions:
        if v.get("isActive"):
            old_cv_acc = v.get("cvAccuracy")
            old_active_version_num = v.get("version", 1)
            break

    # Save current model as version before overwriting
    is_first_model = not model_path(workflow_id).exists()
    
    try:
        next_ver = get_next_version_number(workflow_id)
        if not is_first_model:
            save_current_as_version(workflow_id, next_ver)
            keep_only_last_n_versions(workflow_id, n=5)
    except Exception as e:
        logger.warning(f"Could not save version for '{workflow_id}': {e}")
        next_ver = 1

    # Assign version number to the new model's metadata
    new_version_num = 1 if is_first_model else (next_ver + 1 if next_ver > 0 else 1)
    state.trained_metadata_store[workflow_id]["version"] = new_version_num

    # Save new model
    save_trained_classifier(workflow_id)

    # Compare accuracies and prepare version decision
    version_decision = {
        "autoActivated": True,
        "reason": "New model activated",
        "oldCvAccuracy": old_cv_acc,
        "newCvAccuracy": cv_accuracy,
        "availableVersions": list_versions(workflow_id)
    }

    if old_cv_acc is not None and cv_accuracy is not None:
        if cv_accuracy >= old_cv_acc:
            version_decision["autoActivated"] = True
            version_decision["reason"] = f"New CV ({cv_accuracy:.4f}) is >= old ({old_cv_acc:.4f})"
            logger.info(f"✓ Auto-activated new model: CV {cv_accuracy:.4f} >= {old_cv_acc:.4f}")
        else:
            version_decision["autoActivated"] = False
            version_decision["reason"] = f"New CV ({cv_accuracy:.4f}) is worse than old ({old_cv_acc:.4f})"
            logger.warning(f"⚠ New model CV ({cv_accuracy:.4f}) is worse than old ({old_cv_acc:.4f}) - Swapping backup and active to keep old model active")
            try:
                # Swap model PKL files
                temp_pkl = TRAINED_MODELS_DIR / f"{workflow_id}_temp.pkl"
                current_path = model_path(workflow_id)
                old_model_backup_path = version_path(workflow_id, next_ver)
                new_model_backup_path = version_path(workflow_id, new_version_num)
                
                if current_path.exists() and old_model_backup_path.exists():
                    # Move the new worse model to temp
                    current_path.rename(temp_pkl)
                    
                    # Restore the old better model from its backup to current
                    old_model_backup_path.rename(current_path)
                    
                    # Save the new worse model as its own distinct version number
                    temp_pkl.rename(new_model_backup_path)
                    
                    logger.info(f"✓ Rolled back active PKL file. Worse model saved as v{new_version_num}")

                # Do the same for JSON data
                temp_json = TRAINING_DATA_DIR / f"{workflow_id}_temp.json"
                current_json = training_data_path(workflow_id)
                old_json_backup_path = TRAINING_DATA_DIR / f"{workflow_id}_v{next_ver}.json"
                new_json_backup_path = TRAINING_DATA_DIR / f"{workflow_id}_v{new_version_num}.json"
                
                if current_json.exists() and old_json_backup_path.exists():
                    current_json.rename(temp_json)
                    old_json_backup_path.rename(current_json)
                    temp_json.rename(new_json_backup_path)
                    logger.info(f"✓ Rolled back active JSON file. Worse JSON saved as v{new_version_num}")

                # Reload the rolled-back active model into memory
                load_trained_classifier(workflow_id)
                
                # Explicitly restore the old version number to the rolled-back model
                state.trained_metadata_store[workflow_id]["version"] = old_active_version_num
                
                # And save it back to the PKL so it persists
                with open(current_path, "rb") as src:
                    payload = pickle.load(src)
                if "metadata" not in payload:
                    payload["metadata"] = {}
                payload["metadata"]["version"] = old_active_version_num
                with open(current_path, "wb") as dst:
                    pickle.dump(payload, dst)
                    
            except Exception as swap_err:
                logger.error(f"Failed to rollback/swap model version: {swap_err}")

    # Re-fetch version list after potentially modifying activation states
    version_decision["availableVersions"] = list_versions(workflow_id)

    logger.info(
        f"✓ Training complete for '{workflow_id}'! "
        f"CV={cv_accuracy}, Test={split_eval.get('testAccuracy')}"
    )

    # Add version decision to metadata
    state.trained_metadata_store[workflow_id]["versionDecision"] = version_decision

    return state.trained_metadata_store[workflow_id]


# ─── Full retrain ─────────────────────────────────────────────────────────────

def train_intent_classifier(
    workflow_id: str,
    subIntents: list[SubIntent],
    augmentation_factor: int = 2,
    entities: Optional[list] = None,
) -> dict:
    """
    Full retrain — encodes every subintent from scratch, rebuilds embedding
    cache, runs 80/20 evaluation, fits final classifier on all data.
    """
    if not state.intent_model:
        raise RuntimeError("Sentence encoder is not loaded yet.")

    sentences: list              = []
    labels: list                 = []
    intent_example_counts: dict  = {}
    raw_context_sentences: list  = []
    embeddings_per_intent: dict  = {}

    for sub in subIntents:
        raw_examples = list(sub.examples)
        if sub.label:       raw_examples.append(sub.label)
        if sub.description: raw_examples.append(sub.description)

        if not raw_examples:
            logger.warning(f"  ⚠️  Subintent '{sub.name}' has NO examples — skipping.")
            continue

        raw_context_sentences.extend(raw_examples)
        all_examples = augment_examples(raw_examples, augmentation_factor)
        for ex in all_examples:
            sentences.append(ex)
            labels.append(sub.name)

        intent_example_counts[sub.name] = len(all_examples)
        logger.info(f"  [{workflow_id}] '{sub.name}': {len(raw_examples)} raw → {len(all_examples)} augmented")

    if len(set(labels)) < 2:
        skipped      = [s.name for s in subIntents if not (list(s.examples) or s.label or s.description)]
        has_examples = [s.name for s in subIntents if     (list(s.examples) or s.label or s.description)]
        raise ValueError(
            f"Need at least 2 subintents with examples. "
            f"Got {len(has_examples)} with data: {has_examples}. "
            f"Skipped (no examples): {skipped}."
        )

    logger.info(f"[FULL] '{workflow_id}': {len(sentences)} sentences × {len(set(labels))} intents ...")

    t0 = time.time()
    X  = state.intent_model.encode(sentences, show_progress_bar=True, batch_size=32)
    encode_time = round(time.time() - t0, 2)
    logger.info(f"Encoding done in {encode_time}s — shape: {X.shape}")

    idx = 0
    for sub in subIntents:
        n = intent_example_counts.get(sub.name)
        if n is None:
            continue
        embeddings_per_intent[sub.name] = X[idx: idx + n]
        idx += n
    save_embedding_cache(workflow_id, embeddings_per_intent)
    save_training_data(workflow_id, subIntents, entities or [], augmentation_factor)

    return _fit_classifier_and_evaluate(
        workflow_id           = workflow_id,
        sentences             = sentences,
        labels                = labels,
        X                     = X,
        augmentation_factor   = augmentation_factor,
        entities              = entities,
        subIntents            = subIntents,
        raw_context_sentences = raw_context_sentences,
        encode_time           = encode_time,
        incremental           = False,
    )


# ─── Incremental retrain ──────────────────────────────────────────────────────

def train_intent_classifier_incremental(
    workflow_id: str,
    subIntents: list[SubIntent],
    augmentation_factor: int = 2,
    entities: Optional[list] = None,
) -> dict:
    """
    Incremental retrain — only re-encodes changed subintents, reuses cached
    embeddings for unchanged ones. Runs the same 80/20 evaluation as full retrain.

    Falls back to full encode if no cache exists.
    """
    if not state.intent_model:
        raise RuntimeError("Sentence encoder is not loaded yet.")

    saved_data         = load_training_data(workflow_id)
    saved_fingerprints = compute_intent_fingerprints(saved_data.get("subIntents", [])) if saved_data else {}
    new_fingerprints   = compute_intent_fingerprints(subIntents)

    changed_intents = {
        name for name, fp in new_fingerprints.items()
        if saved_fingerprints.get(name) != fp
    }
    removed_intents = set(saved_fingerprints.keys()) - set(new_fingerprints.keys())

    logger.info(
        f"[INCREMENTAL] '{workflow_id}': "
        f"{len(changed_intents)} changed/new, "
        f"{len(removed_intents)} removed, "
        f"{len(new_fingerprints) - len(changed_intents)} unchanged."
    )

    embedding_cache  = load_embedding_cache(workflow_id)
    cache_hit_count  = 0
    cache_miss_count = 0

    sentences: list              = []
    labels: list                 = []
    intent_example_counts: dict  = {}
    raw_context_sentences: list  = []
    embeddings_per_intent: dict  = {}
    X_parts: list                = []

    for sub in subIntents:
        raw_examples = list(sub.examples)
        if sub.label:       raw_examples.append(sub.label)
        if sub.description: raw_examples.append(sub.description)

        if not raw_examples:
            logger.warning(f"  ⚠️  Subintent '{sub.name}' has NO examples — skipping.")
            continue

        raw_context_sentences.extend(raw_examples)
        all_examples = augment_examples(raw_examples, augmentation_factor)
        intent_example_counts[sub.name] = len(all_examples)

        for ex in all_examples:
            sentences.append(ex)
            labels.append(sub.name)

        s_key = safe_np_key(sub.name)

        if sub.name in changed_intents or s_key not in embedding_cache:
            logger.info(f"  [ENCODE] '{sub.name}' ({len(all_examples)} sentences) — changed or cache miss")
            t_enc    = time.time()
            X_intent = state.intent_model.encode(all_examples, show_progress_bar=False, batch_size=32)
            logger.info(f"    encoded in {round(time.time() - t_enc, 2)}s")
            embeddings_per_intent[sub.name] = X_intent
            X_parts.append(X_intent)
            cache_miss_count += 1
        else:
            cached = embedding_cache[s_key]
            if cached.shape[0] != len(all_examples):
                logger.warning(f"  ⚠️  Cache shape mismatch for '{sub.name}' — re-encoding")
                X_intent = state.intent_model.encode(all_examples, show_progress_bar=False, batch_size=32)
                embeddings_per_intent[sub.name] = X_intent
                X_parts.append(X_intent)
                cache_miss_count += 1
            else:
                logger.info(f"  [CACHE]  '{sub.name}' ({cached.shape[0]} embeddings reused)")
                embeddings_per_intent[sub.name] = cached
                X_parts.append(cached)
                cache_hit_count += 1

    if len(set(labels)) < 2:
        skipped      = [s.name for s in subIntents if not (list(s.examples) or s.label or s.description)]
        has_examples = [s.name for s in subIntents if     (list(s.examples) or s.label or s.description)]
        raise ValueError(
            f"Need at least 2 subintents with examples. "
            f"Got {len(has_examples)} with data: {has_examples}. "
            f"Skipped: {skipped}."
        )

    t0 = time.time()
    X  = np.vstack(X_parts)
    encode_time = round(time.time() - t0, 3)
    logger.info(f"[INCREMENTAL] X={X.shape}, cache hits={cache_hit_count}, misses={cache_miss_count}")

    save_embedding_cache(workflow_id, embeddings_per_intent)
    save_training_data(workflow_id, subIntents, entities or [], augmentation_factor)

    return _fit_classifier_and_evaluate(
        workflow_id           = workflow_id,
        sentences             = sentences,
        labels                = labels,
        X                     = X,
        augmentation_factor   = augmentation_factor,
        entities              = entities,
        subIntents            = subIntents,
        raw_context_sentences = raw_context_sentences,
        encode_time           = encode_time,
        incremental           = True,
        changed_intents       = changed_intents,
        cache_hits            = cache_hit_count,
        cache_misses          = cache_miss_count,
    )
