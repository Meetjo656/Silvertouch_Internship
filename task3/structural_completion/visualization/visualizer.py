import cv2
import numpy as np
from typing import List, Dict, Any

class Visualizer:
    @staticmethod
    def draw_dashed_rectangle(img, pt1, pt2, color, thickness, style='dotted'):
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Draw dashed lines for the 4 edges
        Visualizer._draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness)
        Visualizer._draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness)
        Visualizer._draw_dashed_line(img, (x2, y2), (x1, y2), color, thickness)
        Visualizer._draw_dashed_line(img, (x1, y2), (x1, y1), color, thickness)

    @staticmethod
    def _draw_dashed_line(img, pt1, pt2, color, thickness, dash_length=10):
        x1, y1 = pt1
        x2, y2 = pt2
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        dashes = int(dist / dash_length)
        for i in range(dashes):
            start = (int(x1 + (x2 - x1) * i / dashes), int(y1 + (y2 - y1) * i / dashes))
            end = (int(x1 + (x2 - x1) * (i + 0.5) / dashes), int(y1 + (y2 - y1) * (i + 0.5) / dashes))
            cv2.line(img, start, end, color, thickness)

    @staticmethod
    def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        out_image = image.copy()
        
        for det in detections:
            x_min, y_min, x_max, y_max = [int(v) for v in det["bbox"]]
            source = det.get("source", "detector")
            class_name = det.get("class_name", "Unknown")
            
            if source == "geometry_reasoner":
                color = (255, 0, 0) # Blue (BGR)
                thickness = 3
                Visualizer.draw_dashed_rectangle(out_image, (x_min, y_min), (x_max, y_max), color, thickness)
                
                conf = det.get("confidence", {})
                score = conf.get("score", 0) if isinstance(conf, dict) else conf
                label = f"INFERRED {class_name} {score:.2f}"
                cv2.putText(out_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                color = (0, 255, 0) # Green for detector
                thickness = 2
                cv2.rectangle(out_image, (x_min, y_min), (x_max, y_max), color, thickness)
                
                conf = det.get("confidence", 0)
                score = conf.get("score", 0) if isinstance(conf, dict) else conf
                label = f"{class_name} {score:.2f}"
                cv2.putText(out_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
        return out_image
