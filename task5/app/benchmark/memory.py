import sys
import psutil
import os
from typing import Any

class MemoryBenchmark:
    """Tracks RAM usage and memory footprints of models/caches."""
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak_recorded: float = 0.0
    
    def measure_model_size(self, model: Any) -> float:
        """Measures the approximate in-memory size of a Python object in MB."""
        return sys.getsizeof(model) / (1024 * 1024)
    
    def measure_ram(self) -> float:
        """Returns the current RSS RAM usage of the process in MB."""
        rss = self.process.memory_info().rss / (1024 * 1024)
        if rss > self.peak_recorded:
            self.peak_recorded = rss
        return rss
    
    def measure_cache(self, cache_dict: dict) -> float:
        """Estimates the size of a cache dictionary in MB."""
        size = sys.getsizeof(cache_dict)
        for k, v in cache_dict.items():
            size += sys.getsizeof(k) + sys.getsizeof(v)
        return size / (1024 * 1024)
    
    def peak_memory(self) -> float:
        """Returns the peak memory recorded during the lifecycle of this benchmark."""
        # Also poll current memory just in case
        self.measure_ram()
        return self.peak_recorded
    
    def model_count(self, registry: dict) -> int:
        """Returns the number of models loaded in a registry."""
        return len(registry)
    
    def embedding_cache_size(self, cache: list) -> float:
        """Measures the size of embedded vector caches."""
        return sys.getsizeof(cache) / (1024 * 1024)
    
    def pkl_usage(self, file_path: str) -> float:
        """Returns the size of a pickle file on disk in MB."""
        if os.path.exists(file_path):
            return os.path.getsize(file_path) / (1024 * 1024)
        return 0.0