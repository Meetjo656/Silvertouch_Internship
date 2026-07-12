from typing import List, Dict, Any, Tuple
from ..graph import PanelGraph

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

class RectangleFilter:
    @staticmethod
    def filter_and_feature_extract(raw_rectangles: List[List[int]], graph: PanelGraph, artwork_crop: Tuple[int, int, int, int]) -> List[Dict[str, Any]]:
        """
        Rejects noise and returns a list of candidate dictionaries with rich features.
        """
        valid_candidates = []
        c_xmin, c_ymin, c_xmax, c_ymax = artwork_crop
        
        for r in raw_rectangles:
            x_min, y_min, x_max, y_max = r
            width = x_max - x_min
            height = y_max - y_min
            area = width * height
            
            # Rejection 1: Too small
            if area < 1000 or width < 10 or height < 10:
                continue
                
            # Rejection 2: Unreasonably skinny (likely a line, not a panel)
            aspect_ratio = max(width/height, height/width)
            if aspect_ratio > 15:
                continue
                
            # Rejection 3: Heavily overlaps existing RF-DETR detections
            # We don't want to propose a geometry that's already perfectly detected
            is_redundant = False
            for p in graph.nodes.values():
                if calculate_iou(r, p.bbox) > 0.8:
                    is_redundant = True
                    break
            if is_redundant:
                continue
                
            # Feature Extraction
            center_x = x_min + width / 2.0
            center_y = y_min + height / 2.0
            
            touches_artwork_edge = (
                abs(x_min - c_xmin) < 15 or 
                abs(x_max - c_xmax) < 15 or 
                abs(y_min - c_ymin) < 15 or 
                abs(y_max - c_ymax) < 15
            )
            
            candidate = {
                "bbox": r,
                "features": {
                    "width": width,
                    "height": height,
                    "area": area,
                    "aspect_ratio": aspect_ratio,
                    "center": (center_x, center_y),
                    "touches_artwork_edge": touches_artwork_edge,
                    # We can add touches_other_rectangle later during matching if needed
                }
            }
            valid_candidates.append(candidate)
            
        return valid_candidates
