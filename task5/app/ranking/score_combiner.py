import math
from typing import Dict, List, Tuple, Optional

class ScoreCombiner:
    """
    Combines various ML and heuristic scores (workflow, intent, entity, context)
    into a final confidence score, along with statistical bounds.
    """
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # Default weights if not provided
        self.weights = weights or {
            'workflow': 0.3,
            'intent': 0.4,
            'entity': 0.2,
            'context': 0.1
        }

    def combine(self, workflow_score: float, intent_score: float, entity_score: float, context_score: float) -> float:
        """
        Calculates the final confidence score based on the weighted sum.
        """
        scores = {
            'workflow': workflow_score,
            'intent': intent_score,
            'entity': entity_score,
            'context': context_score
        }
        
        raw_score = self.weighted_sum(scores)
        return self.normalize(raw_score)

    def weighted_sum(self, scores: Dict[str, float]) -> float:
        """Computes the weighted sum of provided scores."""
        total = 0.0
        weight_sum = 0.0
        
        for key, score in scores.items():
            if key in self.weights:
                weight = self.weights[key]
                total += score * weight
                weight_sum += weight
                
        # Handle case where weights sum to 0
        if weight_sum > 0:
            return total / weight_sum
        return 0.0

    def normalize(self, score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Normalizes a score to fit strictly within the allowed bounds (default 0.0 to 1.0)."""
        if score < min_val:
            return min_val
        if score > max_val:
            return max_val
        return score

    def confidence_interval(self, scores: List[float], z_score: float = 1.96) -> Tuple[float, float]:
        """
        Computes a confidence interval for a set of scores (default 95% CI).
        """
        if not scores:
            return (0.0, 0.0)
            
        n = len(scores)
        if n == 1:
            return (scores[0], scores[0])
            
        mean = sum(scores) / n
        variance = sum((x - mean) ** 2 for x in scores) / (n - 1)
        std_dev = math.sqrt(variance)
        
        margin_of_error = z_score * (std_dev / math.sqrt(n))
        
        lower_bound = max(0.0, mean - margin_of_error)
        upper_bound = min(1.0, mean + margin_of_error)
        
        return (lower_bound, upper_bound)

    def uncertainty(self, scores: List[float]) -> float:
        """
        Estimates the uncertainty of a set of scores.
        Using sample standard deviation as a simple proxy for uncertainty.
        """
        if not scores or len(scores) < 2:
            return 0.0
            
        n = len(scores)
        mean = sum(scores) / n
        variance = sum((x - mean) ** 2 for x in scores) / (n - 1)
        return math.sqrt(variance)