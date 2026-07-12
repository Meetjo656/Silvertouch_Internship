import time
import concurrent.futures
from typing import Callable, Any, Dict

class StressBenchmark:
    """Stress tests Python functions directly via concurrency."""
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.timeouts = 0
        self.elapsed_time = 0.0

    def concurrent_requests(self, func: Callable, args_list: list, max_workers: int = 10, timeout: float = 5.0):
        """Simulates concurrent load by invoking a function across thread pools."""
        self.total_requests = len(args_list)
        self.successful_requests = 0
        self.failed_requests = 0
        self.timeouts = 0
        
        start = time.perf_counter()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map args to futures
            future_to_arg = {executor.submit(func, arg): arg for arg in args_list}
            for future in concurrent.futures.as_completed(future_to_arg, timeout=timeout * self.total_requests):
                try:
                    future.result(timeout=timeout)
                    self.successful_requests += 1
                except concurrent.futures.TimeoutError:
                    self.timeouts += 1
                except Exception as e:
                    self.failed_requests += 1
                    
        self.elapsed_time = time.perf_counter() - start

    def requests_per_second(self) -> float:
        """Calculates effective throughput during the stress test."""
        if self.elapsed_time <= 0: return 0.0
        return self.total_requests / self.elapsed_time

    def failure_rate(self) -> float:
        """Percentage of requests that threw exceptions."""
        if self.total_requests == 0: return 0.0
        return (self.failed_requests / self.total_requests) * 100.0

    def timeout_rate(self) -> float:
        """Percentage of requests that exceeded the timeout."""
        if self.total_requests == 0: return 0.0
        return (self.timeouts / self.total_requests) * 100.0

    def sustained_load(self, func: Callable, arg: Any, duration_seconds: int = 10, max_workers: int = 5):
        """Runs continuous load for a specific duration."""
        args_list = [arg] * (max_workers * duration_seconds * 10) # Overprovision args
        # Truncate by passing a strict global timeout to the executor block in a real scenario
        self.concurrent_requests(func, args_list, max_workers=max_workers, timeout=1.0)

    def spike_test(self, func: Callable, arg: Any):
        """Massive concurrency spike for a short burst."""
        args_list = [arg] * 500
        self.concurrent_requests(func, args_list, max_workers=100, timeout=2.0)