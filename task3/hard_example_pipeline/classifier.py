from typing import List, Dict, Any, Tuple
import json
import os

def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    return intersection_area / float(bbox1_area + bbox2_area - intersection_area)

class FailureClassifier:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "configs", "categories.json")
        with open(config_path, "r") as f:
            self.categories = set(json.load(f)["categories"])

    def classify(self, predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]], iou_thresh: float = 0.5) -> List[str]:
        """
        Analyzes the discrepancy between predictions and ground truth.
        Returns a list of assigned failure categories.
        """
        assigned_categories = set()
        
        # Count classes
        gt_counts = {}
        for gt in ground_truth:
            c = gt["class_name"]
            gt_counts[c] = gt_counts.get(c, 0) + 1
            
        pred_counts = {}
        for p in predictions:
            c = p["class_name"]
            pred_counts[c] = pred_counts.get(c, 0) + 1
            if p.get("confidence", 1.0) < 0.5:
                assigned_categories.add("low_confidence")

        # Missing panels
        for c, count in gt_counts.items():
            if pred_counts.get(c, 0) < count:
                cat = f"missing_{c.lower().replace('_panel', '')}"
                if cat in self.categories:
                    assigned_categories.add(cat)

        # Duplicate/False Positives
        for p in predictions:
            matched = False
            for gt in ground_truth:
                if p["class_name"] == gt["class_name"]:
                    if calculate_iou(p["bbox"], gt["bbox"]) > iou_thresh:
                        matched = True
                        break
            if not matched:
                assigned_categories.add("false_positive")
                
        # Check if we have multiple predictions for classes that should only have 1 (Front, Back, Top, Bottom)
        for c in ["Front_Panel", "Back_Panel", "Top_Panel", "Bottom_Panel"]:
            if pred_counts.get(c, 0) > 1:
                assigned_categories.add("duplicate_panels")
                
        # "rotated_cartons", "white_pharma", "technical_dielines" can be added manually during review 
        # or detected by simple heuristics/metadata. We will allow appending them later.

        return list(assigned_categories)
