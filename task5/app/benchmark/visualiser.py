import matplotlib.pyplot as plt
import os
from typing import List, Dict

class Visualizer:
    """Generates charts for benchmark results using matplotlib."""
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def latency_chart(self, p50: float, p95: float, p99: float, filename: str = "latency.png"):
        """Generates a bar chart of latency percentiles."""
        plt.figure(figsize=(8, 5))
        plt.bar(['p50', 'p95', 'p99'], [p50, p95, p99], color=['blue', 'orange', 'red'])
        plt.ylabel('Latency (seconds)')
        plt.title('Latency Percentiles')
        return self.save(filename)

    def memory_chart(self, ram_usage: float, peak_memory: float, filename: str = "memory.png"):
        """Generates a bar chart of memory usage."""
        plt.figure(figsize=(8, 5))
        plt.bar(['Current RAM', 'Peak RAM'], [ram_usage, peak_memory], color=['green', 'darkgreen'])
        plt.ylabel('Memory (MB)')
        plt.title('Memory Usage')
        return self.save(filename)

    def accuracy_chart(self, metrics: Dict[str, float], filename: str = "accuracy.png"):
        """Generates a bar chart of accuracy metrics (e.g. Precision, Recall)."""
        plt.figure(figsize=(8, 5))
        plt.bar(list(metrics.keys()), list(metrics.values()), color='purple')
        plt.ylabel('Score')
        plt.ylim(0, 1.0)
        plt.title('Accuracy Metrics')
        return self.save(filename)

    def comparison_chart(self, old_val: float, new_val: float, metric_name: str, filename: str = "comparison.png"):
        """Generates a chart comparing old vs new system."""
        plt.figure(figsize=(8, 5))
        plt.bar(['Old System', 'New System'], [old_val, new_val], color=['gray', 'blue'])
        plt.ylabel(metric_name)
        plt.title(f'Comparison: {metric_name}')
        return self.save(filename)

    def scalability_chart(self, sizes: List[int], times: List[float], filename: str = "scalability.png"):
        """Generates a line chart showing execution time vs dataset size."""
        plt.figure(figsize=(8, 5))
        plt.plot(sizes, times, marker='o', linestyle='-', color='red')
        plt.xlabel('Dataset Size')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Scalability Curve')
        plt.grid(True)
        return self.save(filename)

    def stress_chart(self, concurrency_levels: List[int], error_rates: List[float], filename: str = "stress.png"):
        """Generates a line chart showing error rate vs concurrency."""
        plt.figure(figsize=(8, 5))
        plt.plot(concurrency_levels, error_rates, marker='x', linestyle='--', color='orange')
        plt.xlabel('Concurrency Level (Threads)')
        plt.ylabel('Error Rate (%)')
        plt.title('Stress Test: Errors vs Concurrency')
        plt.grid(True)
        return self.save(filename)

    def save(self, filename: str) -> str:
        """Saves the current matplotlib figure to disk and clears the canvas."""
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        return filepath