import cProfile
import pstats
import io
import psutil
from typing import Callable, Any, List, Optional

class Profiler:
    """Profiles Python functions for execution time and bottlenecks."""
    def __init__(self):
        self.profiler = cProfile.Profile()
        self.stats: Optional[pstats.Stats] = None
        self._execution_time = 0.0

    def _run_profile(self, func: Callable, *args, **kwargs) -> Any:
        self.profiler.enable()
        result = func(*args, **kwargs)
        self.profiler.disable()
        
        s = io.StringIO()
        self.stats = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        return result

    def profile_training(self, train_func: Callable, *args):
        """Profiles a model training function."""
        return self._run_profile(train_func, *args)
    
    def profile_inference(self, infer_func: Callable, *args):
        """Profiles a model inference function."""
        return self._run_profile(infer_func, *args)

    def hottest_functions(self, limit: int = 10) -> List[str]:
        """Returns the top N functions where the most time is spent."""
        if not self.stats: return []
        self.stats.sort_stats('cumulative')
        
        s = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        stats.print_stats(limit)
        
        lines = s.getvalue().split('\n')
        return [line.strip() for line in lines if line.strip() and not line.startswith(' ')]

    def cpu_usage(self) -> float:
        """Returns the current process CPU usage percentage."""
        return psutil.Process().cpu_percent(interval=0.1)

    def execution_time(self) -> float:
        """Total execution time recorded in the profiler."""
        if not self.stats: 
            return 0.0
        return self.stats.get_stats_profile().total_tt

    def bottlenecks(self, threshold: float = 0.2) -> List[str]:
        """Identifies functions taking more than a specified threshold ratio of total time."""
        if not self.stats: return []
        
        profile = self.stats.get_stats_profile()
        total_time = profile.total_tt
        bottlenecks = []
        for func, func_profile in profile.func_profiles.items():
            if total_time > 0 and (func_profile.cumtime / total_time) > threshold:
                bottlenecks.append(f"{func}: {func_profile.cumtime/total_time:.1%}")
                
        return bottlenecks

    