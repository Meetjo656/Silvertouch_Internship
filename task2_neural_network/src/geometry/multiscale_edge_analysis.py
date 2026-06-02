import cv2
import numpy as np
import math
from pathlib import Path

from src.geometry.artwork_suppressor import ArtworkSuppressor
from src.geometry.edge_detector import EdgeDetector
from src.geometry.line_detector import LineDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MultiScaleEdgeAnalysis:
    """Task 2: Multi-Scale Persistence Analysis."""
    
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.suppressor = ArtworkSuppressor(out_dir)
        self.edge_detector = EdgeDetector(out_dir)
        self.line_detector = LineDetector(out_dir)
        
    def _calculate_angle(self, line):
        dx = line["x2"] - line["x1"]
        dy = line["y2"] - line["y1"]
        angle = math.degrees(math.atan2(dy, dx))
        return angle % 180

    def _angle_diff(self, a1, a2):
        diff = abs(a1 - a2) % 180
        return min(diff, 180 - diff)
        
    def analyze_persistence(self, image_id: str, original_img: np.ndarray, base_regularized_lines: list) -> dict:
        """
        Runs line detection at multiple scales and computes persistence score for the base_regularized_lines.
        """
        logger.info(f"[{image_id}] Running Multi-Scale Edge Analysis...")
        
        scales = [1.0, 0.5, 0.25, 0.125]
        scale_lines = {}
        
        h, w = original_img.shape[:2]
        diag = math.hypot(h, w)
        dist_thresh = 0.02 * diag
        angle_thresh = 10.0
        
        for scale in scales:
            if scale == 1.0:
                # We can just use the base_regularized_lines as the 100% scale lines
                # But to be consistent with detection, we can also re-run detection, or just assume all base lines have 100% presence
                scale_lines[scale] = base_regularized_lines
                continue
                
            # Resize image
            new_h, new_w = int(h * scale), int(w * scale)
            if new_h == 0 or new_w == 0:
                scale_lines[scale] = []
                continue
                
            resized = cv2.resize(original_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Detect lines
            bilateral, opened = self.suppressor.process(f"{image_id}_scale_{scale}", resized)
            edges = self.edge_detector.detect_edges(f"{image_id}_scale_{scale}", bilateral, opened)
            lines = self.line_detector.detect_lines(f"{image_id}_scale_{scale}", edges, resized)
            
            # Project lines back to original scale
            projected = []
            for line in lines:
                projected.append({
                    "x1": line["x1"] / scale,
                    "y1": line["y1"] / scale,
                    "x2": line["x2"] / scale,
                    "y2": line["y2"] / scale
                })
            scale_lines[scale] = projected
            
        persistence_scores = []
        
        for base_line in base_regularized_lines:
            base_angle = self._calculate_angle(base_line)
            bx1, by1, bx2, by2 = base_line["x1"], base_line["y1"], base_line["x2"], base_line["y2"]
            
            score = 1 # Always present at 100%
            
            for scale in [0.5, 0.25, 0.125]:
                matched = False
                for proj_line in scale_lines[scale]:
                    proj_angle = self._calculate_angle(proj_line)
                    if self._angle_diff(base_angle, proj_angle) <= angle_thresh:
                        # Check endpoint distance
                        px1, py1, px2, py2 = proj_line["x1"], proj_line["y1"], proj_line["x2"], proj_line["y2"]
                        
                        # Match points (either p1-b1 and p2-b2 OR p1-b2 and p2-b1)
                        d1 = math.hypot(bx1 - px1, by1 - py1)
                        d2 = math.hypot(bx2 - px2, by2 - py2)
                        
                        d3 = math.hypot(bx1 - px2, by1 - py2)
                        d4 = math.hypot(bx2 - px1, by2 - py1)
                        
                        if (d1 < dist_thresh and d2 < dist_thresh) or (d3 < dist_thresh and d4 < dist_thresh):
                            matched = True
                            break
                            
                if matched:
                    score += 1
                else:
                    break # If it disappears at 50%, it doesn't get points for 25%
                    
            persistence_scores.append(score)
            
        return {
            "scores": persistence_scores,
            "scale_lines": scale_lines
        }
