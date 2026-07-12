from typing import List, Dict, Any
from app.hierarchy.metadata_provider import metadata_provider

class CandidateFilter:
    """
    Pure business logic filter that eliminates predictions 
    not belonging to the current workflow.
    """
    def __init__(self):
        pass
        
    def filter_predictions(self, workflow_id: str, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters a list of intent/subintent predictions based on the allowed candidates for the workflow.
        
        Args:
            workflow_id: The resolved workflow context.
            predictions: The raw top-K predictions from the Global Intent Classifier.
                         Format: [{"name": "IntentName", "confidence": 0.85}, ...]
                         
        Returns:
            A filtered list of predictions that are permitted for the workflow.
        """
        allowed_candidates = metadata_provider.get_allowed_candidates(workflow_id)
        
        # If metadata has no allowed candidates (e.g. welcome-intent might not have strict metadata)
        # we can optionally allow all, or return empty. For strict hierarchy, return only allowed.
        # But if allowed_candidates is empty (meaning no metadata), we might want to return predictions as is
        # or return empty. Let's assume if there are no allowed candidates, we don't filter to avoid breaking
        # legacy fallback workflows.
        if not allowed_candidates:
            return predictions
            
        filtered = []
        for pred in predictions:
            name = pred.get("name")
            if name in allowed_candidates:
                filtered.append(pred)
                
        return filtered

candidate_filter = CandidateFilter()