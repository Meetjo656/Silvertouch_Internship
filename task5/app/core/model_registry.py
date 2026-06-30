"""
app/core/model_registry.py — Centralized registry for trained models.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import app.state as state
from app.config import TRAINED_MODELS_DIR

logger = logging.getLogger("mybot-nlu")


class ModelRegistry:
    """Central registry to track trained NLU models, paths, and metadata."""
    
    def __init__(self, registry_file: Path = TRAINED_MODELS_DIR / "model_registry.json"):
        self.registry_file = registry_file
        self._registry: Dict[str, dict] = self._load_registry()

    def _load_registry(self) -> dict:
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load model registry: {e}")
        return {}

    def _save_registry(self) -> None:
        try:
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(self._registry, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")

    def register_model(self, workflow_id: str, model_path: str, metadata: dict) -> None:
        """Register a model with its metadata and storage path."""
        entry = {
            "workflow_id": workflow_id,
            "workflow_name": metadata.get("workflowName", workflow_id),
            "version": metadata.get("version", 1),
            "accuracy": metadata.get("cvAccuracy") or metadata.get("testAccuracy") or 0.0,
            "created_at": metadata.get("trainedAt", ""),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "classifier": metadata.get("classifier", "LogisticRegression"),
            "training_examples": metadata.get("totalSentences", 0),
            "model_path": str(model_path),
            "full_metadata": metadata
        }
        
        self._registry[workflow_id] = entry
        self._save_registry()
        logger.info(f"Registered model for '{workflow_id}' in ModelRegistry")

    def get_model(self, workflow_id: str) -> Optional[dict]:
        """
        Returns the loaded model dict (classifier, label_encoder, etc) if it's registered.
        Will trigger a lazy load if it's not currently in memory.
        """
        if workflow_id not in self._registry:
            logger.warning(f"Workflow '{workflow_id}' not found in registry.")
            return None
            
        # Is it in memory?
        if workflow_id in state.trained_classifiers:
            return {
                "classifier": state.trained_classifiers[workflow_id],
                "label_encoder": state.trained_label_encoders[workflow_id],
                "metadata": state.trained_metadata_store.get(workflow_id, {}),
                "entity_classifiers": state.trained_entity_classifiers.get(workflow_id, {})
            }
            
        # Otherwise, load it lazily
        try:
            from app.core.persistence import load_trained_classifier
            load_trained_classifier(workflow_id)
            if workflow_id in state.trained_classifiers:
                return {
                    "classifier": state.trained_classifiers[workflow_id],
                    "label_encoder": state.trained_label_encoders[workflow_id],
                    "metadata": state.trained_metadata_store.get(workflow_id, {}),
                    "entity_classifiers": state.trained_entity_classifiers.get(workflow_id, {})
                }
        except Exception as e:
            logger.error(f"Failed to lazy load model '{workflow_id}': {e}")
        return None

    def list_models(self) -> List[dict]:
        """List all registered models' metadata."""
        return list(self._registry.values())

    def remove_model(self, workflow_id: str) -> bool:
        """Remove model from registry."""
        if workflow_id in self._registry:
            del self._registry[workflow_id]
            self._save_registry()
            logger.info(f"Removed '{workflow_id}' from registry.")
            return True
        return False

    def get_best_version(self, workflow_id: str) -> Optional[int]:
        """Return the best version based on accuracy from available files, or active version."""
        from app.core.persistence import list_versions
        versions = list_versions(workflow_id)
        if not versions:
            return None
            
        valid_versions = [v for v in versions if v.get("cvAccuracy") is not None]
        if not valid_versions:
            return versions[0].get("version")
            
        best = max(valid_versions, key=lambda x: x["cvAccuracy"])
        return best["version"]

# Singleton instance
registry = ModelRegistry()
