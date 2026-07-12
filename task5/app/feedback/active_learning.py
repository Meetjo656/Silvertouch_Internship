import json
import os
from datetime import datetime

class ActiveLearning:
    def __init__(self, uncertainty_threshold=0.6, queue_path="retraining_queue.json"):
        self.uncertainty_threshold = uncertainty_threshold
        self.queue_path = queue_path
        self.retraining_queue = self._load_queue()

    def _load_queue(self):
        if os.path.exists(self.queue_path):
            try:
                with open(self.queue_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    def _save_queue(self):
        with open(self.queue_path, "w", encoding="utf-8") as f:
            json.dump(self.retraining_queue, f, indent=4)

    def evaluate_prediction(self, prediction, confidence):
        # Calculate uncertainty and decide if we need to collect or retrain
        unc = self.uncertainty(confidence)
        collect = self.should_collect(unc)
        retrain = self.should_retrain(unc)
        return {
            "uncertainty": unc,
            "should_collect": collect,
            "should_retrain": retrain
        }

    def uncertainty(self, confidence):
        # Simple uncertainty score based on confidence
        # Lower confidence means higher uncertainty
        return 1.0 - confidence

    def should_collect(self, uncertainty_score):
        # Collect if uncertainty is high enough (e.g., confidence is below 0.8)
        # 1.0 - 0.8 = 0.2
        return uncertainty_score > (1.0 - 0.8)

    def should_retrain(self, uncertainty_score):
        # Retrain if uncertainty is very high (e.g., confidence is below our threshold of 0.6)
        # 1.0 - 0.6 = 0.4
        return uncertainty_score > (1.0 - self.uncertainty_threshold)

    def priority_score(self, entry_frequency, uncertainty_score):
        # Calculate priority based on frequency of failure and uncertainty
        # Higher frequency and higher uncertainty means higher priority
        return (entry_frequency * 0.5) + (uncertainty_score * 0.5)

    def build_retraining_queue(self, feedback_entries):
        # Organize feedback entries into a retraining queue based on frequency and priority
        frequency_map = {}
        
        # Calculate frequencies
        for entry in feedback_entries:
            intent = entry.get("actual", {}).get("intent")
            workflow = entry.get("actual", {}).get("workflow")
            if not intent or not workflow:
                continue
            
            key = f"{workflow}::{intent}"
            if key not in frequency_map:
                frequency_map[key] = {
                    "count": 0,
                    "samples": [],
                    "latest_timestamp": entry.get("timestamp")
                }
            frequency_map[key]["count"] += 1
            frequency_map[key]["samples"].append(entry)
            frequency_map[key]["latest_timestamp"] = max(
                frequency_map[key]["latest_timestamp"], 
                entry.get("timestamp", "")
            )

        # Build queue
        new_queue = []
        for key, data in frequency_map.items():
            count = data["count"]
            # Assume average uncertainty for the samples in this bucket
            # For simplicity in this example, we'll assign a base score that scales with count
            # A real implementation would pull the specific confidence/uncertainty of these predictions
            avg_uncertainty = 0.5 
            priority = self.priority_score(count, avg_uncertainty)
            
            priority_label = "LOW"
            if priority > 5:
                priority_label = "HIGH"
            elif priority > 2:
                priority_label = "MEDIUM"

            new_queue.append({
                "target": key,
                "frequency": count,
                "priority_score": priority,
                "priority": priority_label,
                "latest_timestamp": data["latest_timestamp"],
                "reason": "Wrong intent/workflow"
            })

        # Sort by priority score descending
        new_queue.sort(key=lambda x: x["priority_score"], reverse=True)
        
        self.retraining_queue = new_queue
        self._save_queue()
        return self.retraining_queue

    def incremental_train(self, dataset_path):
        # Trigger actual retraining mechanism
        print(f"Triggering incremental training using dataset: {dataset_path}")
        print("This would normally call the model's fit/train method with new epochs.")
        # Stub for the actual training logic
        return True

    def evaluate_improvement(self, metrics_before, metrics_after):
        # Compare metrics before and after retraining
        improvement = {}
        for metric in metrics_before:
            if metric in metrics_after:
                improvement[metric] = metrics_after[metric] - metrics_before[metric]
        return improvement