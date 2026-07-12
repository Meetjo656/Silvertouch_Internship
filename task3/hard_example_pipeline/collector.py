import json
import os
from datetime import datetime
from typing import List, Dict, Any

class HardExampleCollector:
    def __init__(self, output_dir: str = "hard_examples"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.db_path = os.path.join(self.output_dir, "hard_examples_db.jsonl")

    def collect(self, 
                image_path: str, 
                predictions: List[Dict[str, Any]], 
                ground_truth: List[Dict[str, Any]], 
                categories: List[str], 
                model_version: str):
        """
        Stores a failure record to the hard examples database.
        
        predictions / ground_truth format:
        [{"class_name": "Front_Panel", "bbox": [x1, y1, x2, y2], "confidence": 0.95}]
        """
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "image_path": image_path,
            "predictions": predictions,
            "ground_truth": ground_truth,
            "categories": categories,
            "model_version": model_version
        }
        
        with open(self.db_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def load_all(self) -> List[Dict[str, Any]]:
        """
        Loads all hard examples from the database.
        """
        if not os.path.exists(self.db_path):
            return []
            
        records = []
        with open(self.db_path, "r") as f:
            for line in f:
                records.append(json.loads(line.strip()))
        return records
