from typing import List, Dict, Any, Optional

class IntentRanker:
    """
    Ranks the filtered candidates and determines the best intent/subintent match.
    """
    def __init__(self):
        pass
        
    def rank(self, filtered_predictions: List[Dict[str, Any]], threshold: float = 0.30) -> Optional[Dict[str, Any]]:
        """
        Returns the top prediction if it meets the confidence threshold.
        
        Args:
            filtered_predictions: List of predictions output by the CandidateFilter.
            threshold: Minimum confidence required to accept the prediction.
            
        Returns:
            The top prediction dictionary, or None if no prediction meets the threshold
            or if the list is empty.
        """
        if not filtered_predictions:
            return None
            
        # Ensure predictions are sorted by confidence descending
        sorted_preds = sorted(filtered_predictions, key=lambda x: x.get("confidence", 0.0), reverse=True)
        
        top_pred = sorted_preds[0]
        
        # Check against confidence threshold
        if top_pred.get("confidence", 0.0) >= threshold:
            return top_pred
            
        return None

intent_ranker = IntentRanker()
