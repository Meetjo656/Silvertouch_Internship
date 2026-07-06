from typing import Dict, Any, List, Optional

class ConfidenceCalibrator:    
    """
    Calibrates raw scores into standardized confidence levels, and
    determines routing actions (accept, clarify, fallback) based on thresholds.
    """
    def __init__(self, accept_threshold: float = 0.75, clarify_threshold: float = 0.50):
        self.accept_threshold = accept_threshold
        self.clarify_threshold = clarify_threshold

    def fit(self, validation_data: List[Dict[str, Any]]):
        """
        Fits the calibrator using validation data to adjust thresholds dynamically.
        validation_data could be [{"raw_score": 0.8, "is_correct": True}, ...]
        """
        if not validation_data:
            return
            
        correct_scores = [d['raw_score'] for d in validation_data if d.get('is_correct')]
        if correct_scores:
            # Example: Set accept threshold to the 10th percentile of correct scores
            correct_scores.sort()
            idx = max(0, int(len(correct_scores) * 0.10))
            self.accept_threshold = correct_scores[idx]
            
            # Set clarify threshold slightly lower
            self.clarify_threshold = self.accept_threshold * 0.70
            
    def calibrate(self, raw_score: float) -> float:
        """
        Calibrates a raw score. (Currently clamps it between 0 and 1).
        Could be extended with Platt scaling or Isotonic Regression.
        """
        return max(0.0, min(1.0, raw_score))

    def confidence_level(self, score: float) -> str:
        """Returns a qualitative confidence level: 'high', 'medium', or 'low'."""
        calibrated = self.calibrate(score)
        if calibrated >= self.accept_threshold:
            return "high"
        elif calibrated >= self.clarify_threshold:
            return "medium"
        return "low"

    def should_accept(self, score: float) -> bool:
        """Returns True if the score is high enough to confidently accept."""
        return self.calibrate(score) >= self.accept_threshold

    def should_clarify(self, score: float) -> bool:
        """Returns True if the score is ambiguous and needs user clarification."""
        calibrated = self.calibrate(score)
        return self.clarify_threshold <= calibrated < self.accept_threshold

    def should_fallback(self, score: float) -> bool:
        """Returns True if the score is too low and should trigger a fallback."""
        return self.calibrate(score) < self.clarify_threshold

    def dynamic_threshold(self, context_factor: float) -> float:
        """
        Calculates a dynamic acceptance threshold based on context factors.
        (e.g., if the user is frustrated, we might want higher confidence).
        """
        return max(0.0, min(1.0, self.accept_threshold * context_factor))

    def explain(self, score: float) -> str:
        """Returns a human-readable explanation of the confidence routing decision."""
        calibrated = self.calibrate(score)
        level = self.confidence_level(calibrated)
        
        if level == "high":
            return f"Score {calibrated:.2f} >= {self.accept_threshold:.2f} (Accept Threshold). Action: ACCEPT."
        elif level == "medium":
            return f"Score {calibrated:.2f} is between {self.clarify_threshold:.2f} and {self.accept_threshold:.2f}. Action: CLARIFY."
        else:
            return f"Score {calibrated:.2f} < {self.clarify_threshold:.2f} (Clarify Threshold). Action: FALLBACK."
    
    