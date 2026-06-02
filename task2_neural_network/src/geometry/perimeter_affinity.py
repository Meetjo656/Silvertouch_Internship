import cv2
import numpy as np
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PerimeterAffinity:
    """Task 4: Perimeter Affinity Analysis."""
    
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_affinity(self, image_id: str, lines: list, original_img: np.ndarray) -> dict:
        """
        Computes distance from each line to the convex hull of the detected network.
        """
        logger.info(f"[{image_id}] Running Perimeter Affinity Analysis...")
        
        if not lines:
            return {"distances": [], "classifications": [], "hull": []}
            
        # Collect all points
        all_points = []
        for line in lines:
            all_points.append([line["x1"], line["y1"]])
            all_points.append([line["x2"], line["y2"]])
            
        all_points = np.array(all_points, dtype=np.int32)
        hull = cv2.convexHull(all_points)
        
        distances = []
        classifications = []
        
        # Calculate diagonal for dynamic thresholding
        h, w = original_img.shape[:2]
        diag = np.sqrt(h**2 + w**2)
        thresh = 0.05 * diag # 5% of diagonal as threshold for connection
        
        for line in lines:
            p1 = (int(line["x1"]), int(line["y1"]))
            p2 = (int(line["x2"]), int(line["y2"]))
            
            d1 = abs(cv2.pointPolygonTest(hull, p1, True))
            d2 = abs(cv2.pointPolygonTest(hull, p2, True))
            
            min_d = min(d1, d2)
            distances.append(min_d)
            
            if min_d <= thresh:
                classifications.append("PERIMETER_CONNECTED")
            else:
                classifications.append("INTERNAL_ONLY")
                
        return {
            "distances": distances,
            "classifications": classifications,
            "hull": hull
        }
