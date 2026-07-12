from typing import List, Dict, Any, Optional

class IntentRanker:
    """
    Ranks intent predictions based on ML confidence, conversation context,
    valid workflows, and entity matching.
    """
    def __init__(self):
        self.ranked_candidates: List[Dict[str, Any]] = []

    def rank(
        self, 
        predictions: List[Dict[str, Any]], 
        workflow: Optional[str] = None, 
        conversation_context: Optional[Dict[str, Any]] = None, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes the ranking pipeline over a set of ML predictions.
        Example prediction format: [{"intent": "faq", "score": 0.7, "required_entities": [...]}]
        """
        # Deep copy to avoid mutating the original predictions
        self.ranked_candidates = [dict(p) for p in predictions]
        
        self.remove_invalid_candidates(workflow)
        self.apply_context_score(conversation_context)
        self.apply_entity_score(metadata)
        self.normalize_scores()
        
        # Sort by score descending
        self.ranked_candidates.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        return self.ranked_candidates
    def remove_invalid_candidates(self, workflow: Optional[str]):
        """Filters out intents that do not belong to the active workflow (if constrained)."""
        if not workflow or workflow == "global":
            return
            
        valid_candidates = []
        for candidate in self.ranked_candidates:
            c_workflow = candidate.get('workflow_id') or candidate.get('intent', '').split('.')[0]
            if c_workflow == workflow:
                valid_candidates.append(candidate)
        self.ranked_candidates = valid_candidates

    def apply_context_score(self, conversation_context: Optional[Dict[str, Any]]):
        """Boosts the score of intents that align with the ongoing conversational context."""
        if not conversation_context:
            return
            
        current_intent = conversation_context.get('current_intent')
        if not current_intent:
            return
            
        for candidate in self.ranked_candidates:
            if candidate.get('intent') == current_intent:
                # Boost score by 15% for continuing the same intent context
                candidate['score'] = candidate.get('score', 0.0) * 1.15
    
    def apply_entity_score(self, metadata: Optional[Dict[str, Any]]):
        """Boosts intent scores if extracted entities strongly correlate with their requirements."""
        if not metadata:
            return
            
        extracted_entities = metadata.get('extracted_entities', {})
        if not extracted_entities:
            return
            
        for candidate in self.ranked_candidates:
            required = candidate.get('required_entities', [])
            if not required:
                continue
                
            match_count = sum(1 for req in required if req in extracted_entities)
            if match_count > 0:
                # Boost score by 10% per matching required entity
                boost = 1.0 + (0.10 * match_count)
                candidate['score'] = candidate.get('score', 0.0) * boost
    
    def normalize_scores(self):
        """Normalizes all scores so they sum to 1.0."""
        if not self.ranked_candidates:
            return
            
        total_score = sum(max(0.0, c.get('score', 0.0)) for c in self.ranked_candidates)
        if total_score > 0:
            for candidate in self.ranked_candidates:
                candidate['score'] = max(0.0, candidate.get('score', 0.0)) / total_score
        else:
            # Distribute evenly if all scores are 0
            count = len(self.ranked_candidates)
            for candidate in self.ranked_candidates:
                candidate['score'] = 1.0 / count
    
    def top_k(self, k: int = 3) -> List[Dict[str, Any]]:
        """Returns the top K ranked candidates."""
        return self.ranked_candidates[:k]
    
    def best(self) -> Optional[Dict[str, Any]]:
        """Returns the single highest-scoring candidate, or None if empty."""
        if self.ranked_candidates:
            return self.ranked_candidates[0]
        return None

    