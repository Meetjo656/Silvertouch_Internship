from typing import Dict, Any

class ComparisonEngine:
    """Compares metrics between two system baselines."""
    def __init__(self):
        self.old_metrics: Dict[str, Any] = {}
        self.new_metrics: Dict[str, Any] = {}

    def _compare_metric(self, metric_name: str) -> float:
        old_val = self.old_metrics.get(metric_name, 0)
        new_val = self.new_metrics.get(metric_name, 0)
        return self.improvement_percentage(old_val, new_val)

    def compare_latency(self) -> float:
        """Returns improvement % for latency (lower is better)."""
        old_val = self.old_metrics.get('latency', 0)
        new_val = self.new_metrics.get('latency', 0)
        # For latency, lower is better, so old - new
        if old_val == 0: return 0.0
        return ((old_val - new_val) / old_val) * 100.0
    
    def compare_accuracy(self) -> float:
        """Returns improvement % for accuracy (higher is better)."""
        return self._compare_metric('accuracy')
    
    def compare_memory(self) -> float:
        """Returns improvement % for memory (lower is better)."""
        old_val = self.old_metrics.get('memory', 0)
        new_val = self.new_metrics.get('memory', 0)
        if old_val == 0: return 0.0
        return ((old_val - new_val) / old_val) * 100.0

    def compare_model_count(self) -> int:
        """Difference in number of loaded models."""
        return self.new_metrics.get('model_count', 0) - self.old_metrics.get('model_count', 0)
    
    def compare_cache(self) -> float:
        """Returns improvement % for cache size (lower is better)."""
        old_val = self.old_metrics.get('cache_size', 0)
        new_val = self.new_metrics.get('cache_size', 0)
        if old_val == 0: return 0.0
        return ((old_val - new_val) / old_val) * 100.0
    
    def compare_training_time(self) -> float:
        """Returns improvement % for training time (lower is better)."""
        old_val = self.old_metrics.get('training_time', 0)
        new_val = self.new_metrics.get('training_time', 0)
        if old_val == 0: return 0.0
        return ((old_val - new_val) / old_val) * 100.0
    
    def improvement_percentage(self, old_val: float, new_val: float) -> float:
        """Generic improvement calculator (assuming higher is better)."""
        if old_val == 0: return 0.0
        return ((new_val - old_val) / old_val) * 100.0
    
    def summary(self) -> Dict[str, float]:
        """Returns a dict of all comparison improvements."""
        return {
            "latency_improvement_pct": self.compare_latency(),
            "accuracy_improvement_pct": self.compare_accuracy(),
            "memory_improvement_pct": self.compare_memory(),
            "training_time_improvement_pct": self.compare_training_time()
        }