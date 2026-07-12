import math
from typing import List, Dict, Any, Tuple
from ..graph import PanelGraph

def get_shared_edge_length(bbox1: List[float], bbox2: List[float], tolerance: float = 15.0) -> float:
    # Check vertical alignment (left/right neighbors)
    if abs(bbox1[2] - bbox2[0]) <= tolerance or abs(bbox1[0] - bbox2[2]) <= tolerance:
        y_overlap = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
        return y_overlap
    # Check horizontal alignment (top/bottom neighbors)
    if abs(bbox1[3] - bbox2[1]) <= tolerance or abs(bbox1[1] - bbox2[3]) <= tolerance:
        x_overlap = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
        return x_overlap
    return 0.0

class Matcher:
    @staticmethod
    def calculate_match_score(cand_feat: Dict[str, Any], anchor: Any, expected_relation: str) -> float:
        """
        Uses Center Distance, Size Similarity, Shared Edge Length, and Alignment to score a candidate against an anchor.
        """
        cand_bbox = cand_feat["bbox"]
        cx, cy = cand_feat["features"]["center"]
        width = cand_feat["features"]["width"]
        height = cand_feat["features"]["height"]
        
        score = 0.0
        
        # 1. Size Similarity
        if expected_relation in ["left_of", "right_of"]:
            # Heights should match
            if anchor.height > 0:
                score += max(0, 1.0 - abs(height - anchor.height) / anchor.height) * 30
        else:
            # Widths should match
            if anchor.width > 0:
                score += max(0, 1.0 - abs(width - anchor.width) / anchor.width) * 30
                
        # 2. Shared Edge
        shared = get_shared_edge_length(cand_bbox, anchor.bbox)
        if shared > 10:
            score += 40
            
        # 3. Center Distance Penalty (too far is bad)
        dist = math.sqrt((cx - anchor.center_x)**2 + (cy - anchor.center_y)**2)
        if dist < (anchor.width + anchor.height):
            score += 30
            
        return score
