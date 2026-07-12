"""
app/core/persistence.py — Save / load trained classifiers and training-data JSON.
"""

import json
import joblib
import logging
import time
from typing import Optional, List, Dict
from pathlib import Path

from app.config import TRAINED_MODELS_DIR, TRAINING_DATA_DIR
import app.state as state

logger = logging.getLogger("mybot-nlu")


# ─── Classifier pkl files ─────────────────────────────────────────────────────

def model_path(workflow_id: str) -> Path:
    return TRAINED_MODELS_DIR / f"{workflow_id}.joblib"


def version_path(workflow_id: str, version_num: int) -> Path:
    """Get path for a versioned model file."""
    return TRAINED_MODELS_DIR / f"{workflow_id}_v{version_num}.joblib"


def save_trained_classifier(workflow_id: str) -> None:
    payload = {
        "classifier":         state.trained_classifiers[workflow_id],
        "label_encoder":      state.trained_label_encoders[workflow_id],
        "metadata":           state.trained_metadata_store.get(workflow_id, {}),
        "entity_classifiers": state.trained_entity_classifiers.get(workflow_id, {}),
    }
    path = model_path(workflow_id)
    with open(path, "wb") as f:
        joblib.dump(payload, f)
    logger.info(f"Saved trained classifier for '{workflow_id}' → {path}")

    from app.core.model_registry import registry
    registry.register_model(workflow_id, str(path), payload["metadata"])


def load_trained_classifier(workflow_id: str) -> None:
    """Load a persisted classifier into global state."""
    path = model_path(workflow_id)
    with open(path, "rb") as f:
        payload = joblib.load(f)
    state.trained_classifiers[workflow_id]    = payload["classifier"]
    state.trained_label_encoders[workflow_id] = payload["label_encoder"]
    state.trained_metadata_store[workflow_id] = payload.get("metadata", {})
    if "entity_classifiers" in payload:
        state.trained_entity_classifiers[workflow_id] = payload["entity_classifiers"]

def global_model_path() -> Path:
    return TRAINED_MODELS_DIR / "global_shared_classifiers.joblib"

def save_global_classifiers() -> None:
    payload = {
        "intent_classifier": state.shared_intent_classifier,
        "intent_label_encoder": state.shared_intent_label_encoder,
        "workflow_classifier": state.shared_workflow_classifier,
        "workflow_label_encoder": state.shared_workflow_label_encoder,
    }
    path = global_model_path()
    with open(path, "wb") as f:
        joblib.dump(payload, f)
    logger.info(f"Saved global shared classifiers → {path}")

def load_global_classifiers() -> None:
    path = global_model_path()
    if path.exists():
        with open(path, "rb") as f:
            payload = joblib.load(f)
        state.shared_intent_classifier = payload.get("intent_classifier")
        state.shared_intent_label_encoder = payload.get("intent_label_encoder")
        state.shared_workflow_classifier = payload.get("workflow_classifier")
        state.shared_workflow_label_encoder = payload.get("workflow_label_encoder")
        logger.info(f"Loaded global shared classifiers from {path}")


# ─── Version Management ───────────────────────────────────────────────────────

def list_versions(workflow_id: str) -> List[Dict]:
    """
    List all available versions for a workflow.
    Returns list of dicts with: version, cvAccuracy, testAccuracy, timestamp, isActive
    """
    versions = []
    active_metadata = state.trained_metadata_store.get(workflow_id, {})

    # List backup versions
    backup_files = sorted(
        TRAINED_MODELS_DIR.glob(f"{workflow_id}_v*.joblib"),
        key=lambda p: p.stat().st_mtime,
        reverse=True  # Newest first
    )

    max_backup_version = 0
    for f in backup_files:
        try:
            v_num = int(f.stem.split("_v")[1].split(".")[0])
            if v_num > max_backup_version:
                max_backup_version = v_num
        except:
            pass

    # Add current active version
    active_version_num = -1
    if active_metadata:
        active_version_num = active_metadata.get("version", max_backup_version + 1)
        versions.append({
            "version": active_version_num,
            "cvAccuracy": active_metadata.get("cvAccuracy"),
            "testAccuracy": active_metadata.get("testAccuracy"),
            "timestamp": active_metadata.get("trainedAt", ""),
            "isActive": True,
            "filename": f"{workflow_id}.joblib"
        })

    for filepath in backup_files:
        try:
            # Extract version number from filename
            version_num = int(filepath.stem.split("_v")[1].split(".")[0])
            
            # Skip if this is the active version
            if version_num == active_version_num:
                continue

            # Load metadata to get accuracy
            with open(filepath, "rb") as f:
                payload = joblib.load(f)
            metadata = payload.get("metadata", {})

            versions.append({
                "version": version_num,
                "cvAccuracy": metadata.get("cvAccuracy"),
                "testAccuracy": metadata.get("testAccuracy"),
                "timestamp": metadata.get("trainedAt", ""),
                "isActive": False,
                "filename": filepath.name
            })
        except Exception as e:
            logger.warning(f"Could not read version file {filepath}: {e}")

    return versions


def save_current_as_version(workflow_id: str, version_num: int) -> None:
    """
    Save current active model and training data as a versioned backup.
    Copies current .joblib to _v{N}.joblib, and current .json to _v{N}.json
    """
    current_path = model_path(workflow_id)
    if current_path.exists():
        version_path_v = version_path(workflow_id, version_num)
        with open(current_path, "rb") as src:
            with open(version_path_v, "wb") as dst:
                dst.write(src.read())
        logger.info(f"Saved model version {version_num} for '{workflow_id}' → {version_path_v}")
    
    current_json = training_data_path(workflow_id)
    if current_json.exists():
        version_json_path = TRAINING_DATA_DIR / f"{workflow_id}_v{version_num}.json"
        with open(current_json, "r", encoding="utf-8") as src:
            with open(version_json_path, "w", encoding="utf-8") as dst:
                dst.write(src.read())
        logger.info(f"Saved JSON version {version_num} for '{workflow_id}' → {version_json_path}")


def keep_only_last_n_versions(workflow_id: str, n: int = 5) -> None:
    """
    Keep only the last N versions of model and training data, delete older ones.
    Always keeps version 0 (current active).
    """
    backup_files = sorted(
        TRAINED_MODELS_DIR.glob(f"{workflow_id}_v*.joblib"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    # Keep first N-1 versions (since we have current + N-1 backups = N total)
    files_to_delete = backup_files[n-1:]

    for filepath in files_to_delete:
        try:
            filepath.unlink()
            logger.info(f"Deleted old version model: {filepath.name}")
            
            # Delete corresponding JSON file if exists
            try:
                version_num = int(filepath.stem.split("_v")[1].split(".")[0])
                version_json_path = TRAINING_DATA_DIR / f"{workflow_id}_v{version_num}.json"
                if version_json_path.exists():
                    version_json_path.unlink()
                    logger.info(f"Deleted old version JSON: {version_json_path.name}")
            except Exception as json_err:
                logger.warning(f"Could not delete JSON for old version: {json_err}")
        except Exception as e:
            logger.warning(f"Could not delete {filepath}: {e}")


def load_version(workflow_id: str, version_num: int) -> None:
    """
    Load a specific version into memory (doesn't activate it).
    """
    if version_num == 0:
        load_trained_classifier(workflow_id)
    else:
        version_path_v = version_path(workflow_id, version_num)
        if not version_path_v.exists():
            raise FileNotFoundError(f"Version {version_num} not found for '{workflow_id}'")

        with open(version_path_v, "rb") as f:
            payload = joblib.load(f)

        state.trained_classifiers[workflow_id]    = payload["classifier"]
        state.trained_label_encoders[workflow_id] = payload["label_encoder"]
        state.trained_metadata_store[workflow_id] = payload.get("metadata", {})
        if "entity_classifiers" in payload:
            state.trained_entity_classifiers[workflow_id] = payload["entity_classifiers"]


def activate_version(workflow_id: str, version_num: int) -> None:
    """
    Activate a specific version by copying it to become the current model and training data.
    """
    if version_num == 0:
        logger.info(f"Version {version_num} is already active for '{workflow_id}'")
        return

    version_path_v = version_path(workflow_id, version_num)
    current_path = model_path(workflow_id)

    if not version_path_v.exists():
        raise FileNotFoundError(f"Version {version_num} not found for '{workflow_id}'")

    # Back up the current active model before overwriting
    if current_path.exists():
        try:
            with open(current_path, "rb") as cur_f:
                cur_payload = joblib.load(cur_f)
            cur_version = cur_payload.get("metadata", {}).get("version", 1)
            
            cur_backup_path = version_path(workflow_id, cur_version)
            if not cur_backup_path.exists():
                with open(current_path, "rb") as src:
                    with open(cur_backup_path, "wb") as dst:
                        dst.write(src.read())
                
                # Back up JSON too
                cur_json_path = training_data_path(workflow_id)
                cur_json_backup_path = TRAINING_DATA_DIR / f"{workflow_id}_v{cur_version}.json"
                if cur_json_path.exists() and not cur_json_backup_path.exists():
                    with open(cur_json_path, "r", encoding="utf-8") as src:
                        with open(cur_json_backup_path, "w", encoding="utf-8") as dst:
                            dst.write(src.read())
        except Exception as e:
            logger.warning(f"Could not back up current model before activating v{version_num}: {e}")

    # Copy version to current model, but explicitly update the version number in its metadata
    with open(version_path_v, "rb") as src:
        payload = joblib.load(src)
        
    if "metadata" not in payload:
        payload["metadata"] = {}
    payload["metadata"]["version"] = version_num
    
    with open(current_path, "wb") as dst:
        joblib.dump(payload, dst)

    # Copy version to current training data json
    version_json_path = TRAINING_DATA_DIR / f"{workflow_id}_v{version_num}.json"
    current_json = training_data_path(workflow_id)
    if version_json_path.exists():
        try:
            with open(version_json_path, "r", encoding="utf-8") as src:
                with open(current_json, "w", encoding="utf-8") as dst:
                    dst.write(src.read())
            logger.info(f"Restored JSON version {version_num} for '{workflow_id}'")
        except Exception as e:
            logger.warning(f"Could not restore JSON version {version_num} for '{workflow_id}': {e}")

    # Load into memory
    load_trained_classifier(workflow_id)

    # Explicitly set the version number in memory so list_versions knows which one is active
    state.trained_metadata_store[workflow_id]["version"] = version_num

    logger.info(f"Activated version {version_num} for '{workflow_id}'")


def delete_version(workflow_id: str, version_num: int) -> None:
    """
    Delete a specific version (cannot delete current active version).
    """
    if version_num == 0:
        raise ValueError("Cannot delete active version (0)")

    version_path_v = version_path(workflow_id, version_num)
    if version_path_v.exists():
        version_path_v.unlink()
        logger.info(f"Deleted version {version_num} model for '{workflow_id}'")
    else:
        raise FileNotFoundError(f"Version {version_num} not found")

    # Delete json version if exists
    version_json_path = TRAINING_DATA_DIR / f"{workflow_id}_v{version_num}.json"
    if version_json_path.exists():
        try:
            version_json_path.unlink()
            logger.info(f"Deleted version {version_num} JSON for '{workflow_id}'")
        except Exception as e:
            logger.warning(f"Could not delete JSON version {version_num}: {e}")


def get_next_version_number(workflow_id: str) -> int:
    """
    Get the next available version number for a workflow.
    """
    backup_files = list(TRAINED_MODELS_DIR.glob(f"{workflow_id}_v*.joblib"))
    if not backup_files:
        return 1

    # Extract version numbers and find max
    version_nums = []
    for filepath in backup_files:
        try:
            version_num = int(filepath.stem.split("_v")[1].split(".")[0])
            version_nums.append(version_num)
        except (IndexError, ValueError):
            continue

    return max(version_nums) + 1 if version_nums else 1


# ─── Training-data JSON ───────────────────────────────────────────────────────

def training_data_path(workflow_id: str) -> Path:
    return TRAINING_DATA_DIR / f"{workflow_id}.json"


def save_training_data(
    workflow_id: str,
    subIntents: list,
    entities: list,
    augmentation_factor: int,
) -> None:
    """Persist the raw (pre-augmentation) training payload to disk."""
    def _get(obj, attr, default):
        return getattr(obj, attr, obj.get(attr, default) if isinstance(obj, dict) else default)

    data = {
        "workflowId":        workflow_id,
        "augmentationFactor": augmentation_factor,
        "subIntents": [
            {
                "name":             _get(s, "name", ""),
                "label":            _get(s, "label", ""),
                "description":      _get(s, "description", ""),
                "examples":         list(_get(s, "examples", []) or []),
                "requiredEntities": list(_get(s, "requiredEntities", []) or []),
            }
            for s in subIntents
        ],
        "entities": [
            {
                "name":     _get(e, "name", ""),
                "type":     _get(e, "type", "regex"),
                "regex":    _get(e, "regex", ""),
                "examples": list(_get(e, "examples", []) or []),
            }
            for e in (entities or [])
        ],
    }
    with open(training_data_path(workflow_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved training data for '{workflow_id}' → {training_data_path(workflow_id)}")


def load_training_data(workflow_id: str) -> Optional[dict]:
    """Load the last-saved raw training data for a workflow, or None if missing."""
    path = training_data_path(workflow_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not read training data for '{workflow_id}': {e}")
        return None
