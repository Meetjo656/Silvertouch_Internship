import os
import json
from datetime import datetime

class FailureReporter:
    def __init__(self, output_dir: str = "failure_reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def log_failure(self, image_id: str, missing_class: str, reason: str, metadata: dict = None):
        """
        Logs a failure to recover a panel to a JSONL file.
        """
        log_file = os.path.join(self.output_dir, "unrecovered_panels.jsonl")
        
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "image_id": image_id,
            "missing_class": missing_class,
            "reason": reason,
            "metadata": metadata or {}
        }
        
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
