import cv2
import numpy as np
import json
import math
from pathlib import Path
from shapely.geometry import LineString
from src.utils.logger import get_logger

logger = get_logger(__name__)

class LineRegularizer:
    """Stage 2.5: Line Regularization. Converts noisy CV lines into CAD geometry."""
    
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def _calculate_angle(self, line):
        dx = line["x2"] - line["x1"]
        dy = line["y2"] - line["y1"]
        angle = math.degrees(math.atan2(dy, dx))
        return angle % 180 # 0 to 180

    def _orthogonalize(self, lines, angle_thresh=5.0):
        """Snaps near-horizontal and near-vertical lines."""
        regularized = []
        for line in lines:
            x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
            angle = self._calculate_angle(line)
            
            # Snap to horizontal (0 or 180)
            if angle <= angle_thresh or angle >= 180 - angle_thresh:
                y_avg = (y1 + y2) // 2
                regularized.append({"x1": x1, "y1": y_avg, "x2": x2, "y2": y_avg})
            # Snap to vertical (90)
            elif abs(angle - 90) <= angle_thresh:
                x_avg = (x1 + x2) // 2
                regularized.append({"x1": x_avg, "y1": y1, "x2": x_avg, "y2": y2})
            else:
                regularized.append(line)
        return regularized
        
    def _merge_collinear(self, lines, dist_thresh=20, angle_thresh=5.0):
        """Merges overlapping collinear segments."""
        # A simple clustering approach based on angle and distance.
        # This is a basic implementation; robust collinear merging is a complex CAD problem.
        if not lines: return []
        
        merged = []
        used = [False] * len(lines)
        
        for i in range(len(lines)):
            if used[i]: continue
            current_line = lines[i]
            used[i] = True
            
            cluster = [current_line]
            angle_i = self._calculate_angle(current_line)
            
            # Form line equation A, B, C for current line: Ax + By + C = 0
            x1, y1, x2, y2 = current_line["x1"], current_line["y1"], current_line["x2"], current_line["y2"]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            norm = math.hypot(A, B)
            if norm == 0: continue
            
            for j in range(i + 1, len(lines)):
                if used[j]: continue
                test_line = lines[j]
                angle_j = self._calculate_angle(test_line)
                
                # Check parallel
                angle_diff = min(abs(angle_i - angle_j), 180 - abs(angle_i - angle_j))
                if angle_diff <= angle_thresh:
                    # Check distance from test line points to current line
                    tx1, ty1, tx2, ty2 = test_line["x1"], test_line["y1"], test_line["x2"], test_line["y2"]
                    d1 = abs(A * tx1 + B * ty1 + C) / norm
                    d2 = abs(A * tx2 + B * ty2 + C) / norm
                    
                    if max(d1, d2) <= dist_thresh:
                        cluster.append(test_line)
                        used[j] = True
                        
            # Project all points in cluster to the line and take extremes
            points = []
            for l in cluster:
                points.append((l["x1"], l["y1"]))
                points.append((l["x2"], l["y2"]))
                
            # Find the two points furthest apart to form the merged line
            max_dist = -1
            best_pair = (points[0], points[1])
            for p1 in points:
                for p2 in points:
                    d = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
                    if d > max_dist:
                        max_dist = d
                        best_pair = (p1, p2)
                        
            merged.append({
                "x1": int(best_pair[0][0]), "y1": int(best_pair[0][1]),
                "x2": int(best_pair[1][0]), "y2": int(best_pair[1][1])
            })
            
        return merged
        
    def _snap_endpoints(self, lines, snap_thresh=25):
        """Snaps endpoints that are very close to each other."""
        endpoints = []
        for i, line in enumerate(lines):
            endpoints.append([line["x1"], line["y1"]])
            endpoints.append([line["x2"], line["y2"]])
            
        endpoints = np.array(endpoints)
        if len(endpoints) == 0: return lines
        
        # Simple greedy clustering
        snapped = np.copy(endpoints)
        used = [False] * len(endpoints)
        
        for i in range(len(endpoints)):
            if used[i]: continue
            cluster_indices = [i]
            used[i] = True
            
            for j in range(i + 1, len(endpoints)):
                if not used[j]:
                    dist = np.linalg.norm(endpoints[i] - endpoints[j])
                    if dist <= snap_thresh:
                        cluster_indices.append(j)
                        used[j] = True
                        
            if len(cluster_indices) > 1:
                # Average point
                avg_pt = np.mean(endpoints[cluster_indices], axis=0).astype(int)
                for idx in cluster_indices:
                    snapped[idx] = avg_pt
                    
        # Reconstruct lines
        regularized = []
        for i in range(len(lines)):
            p1 = snapped[2*i]
            p2 = snapped[2*i + 1]
            if np.linalg.norm(p1 - p2) > 0: # Avoid zero-length lines
                regularized.append({"x1": int(p1[0]), "y1": int(p1[1]), "x2": int(p2[0]), "y2": int(p2[1])})
                
        return regularized
        
    def regularize(self, image_id: str, raw_lines: list, original_img: np.ndarray) -> list:
        logger.info(f"[{image_id}] Running Line Regularization...")
        
        # 1. Orthogonalization
        ortho = self._orthogonalize(raw_lines)
        
        # 2. Merge Collinear
        merged = self._merge_collinear(ortho)
        
        # 3. Snap Endpoints / Gap Repair
        snapped = self._snap_endpoints(merged)
        
        # Draw overlay
        h, w = original_img.shape[:2] if original_img is not None else (1000, 1000)
        overlay = original_img.copy() if original_img is not None else np.zeros((h, w, 3), dtype=np.uint8)
        
        for line in snapped:
            cv2.line(overlay, (line["x1"], line["y1"]), (line["x2"], line["y2"]), (255, 0, 0), 3)
            # Draw endpoints
            cv2.circle(overlay, (line["x1"], line["y1"]), 4, (0, 255, 0), -1)
            cv2.circle(overlay, (line["x2"], line["y2"]), 4, (0, 255, 0), -1)
            
        out_png = self.out_dir / f"{image_id}_regularized_lines.png"
        cv2.imwrite(str(out_png), overlay)
        
        out_json = self.out_dir / f"{image_id}_regularized_lines.json"
        with open(out_json, "w") as f:
            json.dump({"lines": snapped}, f, indent=4)
            
        logger.info(f"[{image_id}] Regularized to {len(snapped)} structural lines.")
        return snapped
