from collections import Counter
from typing import List, Dict, Any, Tuple

class AccuracyBenchmark:
    """Computes standard ML accuracy and conversational metrics."""
    def __init__(self):
        self.results = {}
    
    def evaluate(self, y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
        """Runs a standard evaluation suite."""
        self.results = {
            "precision": self.precision(y_true, y_pred),
            "recall": self.recall(y_true, y_pred),
            "f1": self.f1(y_true, y_pred)
        }
        return self.results
    
    def confusion_matrix(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Dict[str, int]]:
        """Computes a simple confusion matrix."""
        matrix = {}
        for t, p in zip(y_true, y_pred):
            if t not in matrix:
                matrix[t] = {}
            matrix[t][p] = matrix[t].get(p, 0) + 1
        return matrix

    def precision(self, y_true: List[str], y_pred: List[str]) -> float:
        """Micro-averaged precision."""
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return correct / len(y_pred) if y_pred else 0.0
    
    def recall(self, y_true: List[str], y_pred: List[str]) -> float:
        """Micro-averaged recall (same as precision for single-label multi-class)."""
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return correct / len(y_true) if y_true else 0.0
    
    def f1(self, y_true: List[str], y_pred: List[str]) -> float:
        """Micro-averaged F1 score."""
        p = self.precision(y_true, y_pred)
        r = self.recall(y_true, y_pred)
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    
    def top_k_accuracy(self, y_true: List[str], y_pred_list: List[List[str]]) -> float:
        """Checks if the true label is within the top-K predictions."""
        if not y_true or not y_pred_list: return 0.0
        correct = sum(1 for t, p_list in zip(y_true, y_pred_list) if t in p_list)
        return correct / len(y_true)
    
    def workflow_accuracy(self, y_true_wf: List[str], y_pred_wf: List[str]) -> float:
        """Accuracy specifically for the high-level workflow classifier."""
        return self.precision(y_true_wf, y_pred_wf)
    
    def entity_accuracy(self, expected_entities: List[Dict], extracted_entities: List[Dict]) -> float:
        """Computes extraction accuracy (exact match of entity dicts)."""
        if not expected_entities: return 0.0
        correct = sum(1 for e, p in zip(expected_entities, extracted_entities) if e == p)
        return correct / len(expected_entities)

    def fallback_accuracy(self, should_fallback: List[bool], did_fallback: List[bool]) -> float:
        """Measures how accurately the system triggers fallback when it should."""
        correct = sum(1 for e, p in zip(should_fallback, did_fallback) if e == p)
        return correct / len(should_fallback) if should_fallback else 0.0