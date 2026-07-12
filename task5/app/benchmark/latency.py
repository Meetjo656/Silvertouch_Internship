import time
from typing import Callable, Any, List

class LatencyBenchmark:
    """Measures execution speed, throughput, and percentiles."""
    def __init__(self):
        self.latencies: List[float] = []
    
    def measure_single_request(self, func: Callable, *args, **kwargs) -> float:
        """Measures the latency of a single function call."""
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        duration = end - start
        self.latencies.append(duration)
        return duration
    
    def measure_batch(self, func: Callable, batch_args: List[Any]) -> List[float]:
        """Measures latency for a batch of requests sequentially."""
        return [self.measure_single_request(func, arg) for arg in batch_args]
    
    def average_latency(self) -> float:
        """Returns the mean latency."""
        if not self.latencies: return 0.0
        return sum(self.latencies) / len(self.latencies)
    
    def _percentile(self, p: float) -> float:
        if not self.latencies: return 0.0
        sorted_lats = sorted(self.latencies)
        idx = int((p / 100.0) * len(sorted_lats))
        return sorted_lats[min(idx, len(sorted_lats)-1)]

    def percentile_95(self) -> float:
        """Returns the 95th percentile latency."""
        return self._percentile(95)
    
    def percentile_99(self) -> float:
        """Returns the 99th percentile latency."""
        return self._percentile(99)
    
    def warmup(self, func: Callable, *args, iterations: int = 10):
        """Runs the function multiple times without recording to warm up caches."""
        for _ in range(iterations):
            func(*args)
    
    def cold_start(self, func: Callable, *args) -> float:
        """Measures the very first request (cold start)."""
        self.latencies.clear() # Reset to ensure it's recorded fresh
        return self.measure_single_request(func, *args)
    
    def throughput(self, duration_seconds: float) -> float:
        """Calculates requests per second based on recorded latencies within a time window."""
        if not self.latencies or duration_seconds <= 0: return 0.0
        # If we ran len(latencies) requests in duration_seconds
        return len(self.latencies) / duration_seconds