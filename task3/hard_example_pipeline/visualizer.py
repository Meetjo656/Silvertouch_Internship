import cv2
import os
import numpy as np
from typing import List, Dict, Any

class HardExampleVisualizer:
    def __init__(self, output_dir: str = "hard_examples/visualizations"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def draw_composite(self, image_path: str, predictions: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]], categories: List[str]) -> str:
        """
        Draws ground truth in blue, predictions in green, and adds category text.
        """
        if not os.path.exists(image_path):
            return ""
            
        img = cv2.imread(image_path)
        if img is None:
            return ""
            
        # Draw ground truth (Blue)
        for gt in ground_truth:
            x1, y1, x2, y2 = [int(v) for v in gt["bbox"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"GT: {gt['class_name']}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
        # Draw predictions (Green)
        for p in predictions:
            x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            conf = p.get("confidence", 0)
            cv2.putText(img, f"Pred: {p['class_name']} {conf:.2f}", (x1, min(img.shape[0], y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Add summary text at top left
        y_offset = 30
        cv2.putText(img, "Hard Example Mining Review", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y_offset += 30
        for cat in categories:
            cv2.putText(img, f"- {cat}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 25
            
        basename = os.path.basename(image_path)
        out_path = os.path.join(self.output_dir, f"hard_ex_{basename}")
        cv2.imwrite(out_path, img)
        return out_path
