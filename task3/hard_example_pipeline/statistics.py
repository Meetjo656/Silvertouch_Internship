from typing import List, Dict, Any
from collections import Counter
from .collector import HardExampleCollector

class StatisticsGenerator:
    def __init__(self, collector: HardExampleCollector):
        self.collector = collector

    def compute_stats(self) -> Dict[str, Any]:
        records = self.collector.load_all()
        
        if not records:
            return {}

        total_examples = len(records)
        category_counts = Counter()
        
        # We also want to compute average confidence of predictions in failed examples
        total_conf = 0.0
        conf_count = 0
        
        for r in records:
            for cat in r.get("categories", []):
                category_counts[cat] += 1
                
            for p in r.get("predictions", []):
                total_conf += p.get("confidence", 0)
                conf_count += 1
                
        avg_confidence = total_conf / max(1, conf_count)
        
        return {
            "total_examples": total_examples,
            "category_distribution": dict(category_counts),
            "average_confidence": avg_confidence,
            "most_common_failure": category_counts.most_common(1)[0][0] if category_counts else "None"
        }
