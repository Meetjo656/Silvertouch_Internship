import math
from typing import Callable, Any, Dict, List

class ScalabilityBenchmark:
    """Measures system performance scaling across varying dataset sizes."""
    def __init__(self):
        self.results: Dict[str, float] = {}

    def _run_dataset(self, func: Callable, dataset: List[Any], name: str) -> float:
        import time
        start = time.perf_counter()
        func(dataset)
        duration = time.perf_counter() - start
        self.results[name] = duration
        return duration

    def benchmark_small_dataset(self, func: Callable, dataset: List[Any]) -> float:
        return self._run_dataset(func, dataset, "small")
    
    def benchmark_medium_dataset(self, func: Callable, dataset: List[Any]) -> float:
        return self._run_dataset(func, dataset, "medium")
    
    def benchmark_large_dataset(self, func: Callable, dataset: List[Any]) -> float:
        return self._run_dataset(func, dataset, "large")
    
    def benchmark_huge_dataset(self, func: Callable, dataset: List[Any]) -> float:
        return self._run_dataset(func, dataset, "huge")

    def growth_curve(self, sizes: Dict[str, int]) -> str:
        """Determines if scaling is O(1), O(N), O(N^2), etc., based on recorded results."""
        if len(self.results) < 2: return "unknown"
        
        # Simple heuristic to check linearity
        try:
            ratios = []
            keys = ["small", "medium", "large", "huge"]
            available = [k for k in keys if k in self.results and k in sizes]
            
            for i in range(1, len(available)):
                prev, curr = available[i-1], available[i]
                size_ratio = sizes[curr] / float(sizes[prev])
                time_ratio = self.results[curr] / self.results[prev]
                ratios.append(time_ratio / size_ratio)
                
            avg_ratio = sum(ratios) / len(ratios)
            if avg_ratio < 1.1: return "O(N) - Linear"
            elif avg_ratio < 2.0: return "O(N log N) - Log-Linear"
            else: return "O(N^2) - Quadratic or Worse"
        except Exception:
            return "error calculating curve"
    
    def estimate_future(self, current_size: int, current_time: float, future_size: int, curve: str) -> float:
        """Projects future execution time based on the identified curve."""
        ratio = future_size / float(current_size)
        if "Linear" in curve:
            return current_time * ratio
        elif "Log-Linear" in curve:
            return current_time * ratio * math.log2(ratio) if ratio > 1 else current_time
        elif "Quadratic" in curve:
            return current_time * (ratio ** 2)
        return current_time * ratio