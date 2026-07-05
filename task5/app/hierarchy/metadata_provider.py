from typing import List, Dict, Optional
from app.core.persistence import load_training_data
import logging

logger = logging.getLogger("mybot-nlu")

class MetadataProvider:
    """
    Source of truth for hierarchical relationships.
    Loads data from the training JSON registry to derive:
    Workflow -> Allowed SubIntents -> Required Entities
    """
    
    def __init__(self):
        # We can cache metadata if needed to avoid disk reads on every request.
        self._cache: Dict[str, dict] = {}
        
    def _load_metadata(self, workflow_id: str) -> dict:
        if workflow_id not in self._cache:
            data = load_training_data(workflow_id)
            if not data:
                logger.warning(f"No training data found for workflow '{workflow_id}'")
                self._cache[workflow_id] = {}
            else:
                self._cache[workflow_id] = data
        return self._cache[workflow_id]
        
    def get_allowed_candidates(self, workflow_id: str) -> List[str]:
        """
        Returns a list of allowed subintent/intent names for a given workflow.
        """
        data = self._load_metadata(workflow_id)
        if not data:
            return []
            
        sub_intents = data.get("subIntents", [])
        # Extract the names of the allowed subintents
        return [sub.get("name", "") for sub in sub_intents if sub.get("name")]
        
    def get_required_entities(self, workflow_id: str, candidate_name: str) -> List[dict]:
        """
        Returns the required entities for a specific candidate within a workflow.
        """
        data = self._load_metadata(workflow_id)
        sub_intents = data.get("subIntents", [])
        
        for sub in sub_intents:
            if sub.get("name") == candidate_name:
                return sub.get("requiredEntities", [])
                
        return []

    def clear_cache(self, workflow_id: Optional[str] = None):
        """Clear the metadata cache, useful if training data gets updated."""
        if workflow_id and workflow_id in self._cache:
            del self._cache[workflow_id]
        elif not workflow_id:
            self._cache.clear()

# Singleton instance
metadata_provider = MetadataProvider()
