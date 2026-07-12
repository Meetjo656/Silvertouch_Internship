import cv2
import numpy as np
import math
from pathlib import Path
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points
from src.utils.logger import get_logger

logger = get_logger(__name__)

class GapBridgeSolver:
    """Stage 6.3: Intelligently bridges massive gaps to close the topology."""
    
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def solve_gaps(self, image_id: str, pslg_edges: list, silhouette_poly: Polygon, original_img: np.ndarray) -> list:
        logger.info(f"[{image_id}] Running Gap Bridge Solver...")
        
        # 1. Identify dangling endpoints
        endpoint_degrees = {}
        def pt_key(coord): return (round(coord[0], 2), round(coord[1], 2))
        
        for edge in pslg_edges:
            coords = list(edge.coords)
            if len(coords) < 2: continue
            p1, p2 = pt_key(coords[0]), pt_key(coords[-1])
            endpoint_degrees[p1] = endpoint_degrees.get(p1, 0) + 1
            endpoint_degrees[p2] = endpoint_degrees.get(p2, 0) + 1
            
        dangling = [pt for pt, deg in endpoint_degrees.items() if deg == 1]
        
        # 2. Generate Candidate Bridges
        new_bridges = []
        max_bridge_dist = 250 # Can bridge up to 250px gaps
        
        for pt in dangling:
            shapely_pt = Point(pt)
            best_bridge = None
            best_score = float('inf')
            
            # Candidate A: Bridge to the Silhouette Fence
            if silhouette_poly is not None and not silhouette_poly.is_empty:
                fence_boundary = silhouette_poly.exterior
                _, nearest_fence_pt = nearest_points(shapely_pt, fence_boundary)
                dist_to_fence = shapely_pt.distance(nearest_fence_pt)
                
                if dist_to_fence < max_bridge_dist:
                    # Silhouette bridging is highly prioritized
                    score = dist_to_fence * 0.5 
                    if score < best_score:
                        best_score = score
                        best_bridge = LineString([shapely_pt, nearest_fence_pt])
                        
            # Candidate B: Bridge to another dangling endpoint
            for other_pt in dangling:
                if pt == other_pt: continue
                other_shapely = Point(other_pt)
                dist = shapely_pt.distance(other_shapely)
                
                if dist < max_bridge_dist:
                    # Check if orthogonal/horizontal (angle score)
                    dx = abs(pt[0] - other_pt[0])
                    dy = abs(pt[1] - other_pt[1])
                    angle = math.degrees(math.atan2(dy, dx))
                    
                    angle_penalty = 1.0
                    if angle < 10 or angle > 80: # Close to horizontal or vertical
                        angle_penalty = 0.8
                        
                    score = dist * angle_penalty
                    if score < best_score:
                        best_score = score
                        best_bridge = LineString([shapely_pt, other_shapely])
                        
            if best_bridge is not None:
                new_bridges.append(best_bridge)
                
        # Draw overlay
        h, w = original_img.shape[:2] if original_img is not None else (1000, 1000)
        overlay = original_img.copy() if original_img is not None else np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw original edges
        for edge in pslg_edges:
            ext_coords = np.array(edge.coords, dtype=np.int32)
            cv2.polylines(overlay, [ext_coords], isClosed=False, color=(255, 255, 255), thickness=1)
            
        # Draw bridges
        for bridge in new_bridges:
            ext_coords = np.array(bridge.coords, dtype=np.int32)
            cv2.polylines(overlay, [ext_coords], isClosed=False, color=(0, 165, 255), thickness=3) # Orange
            
        out_png = self.out_dir / f"{image_id}_bridge_candidates.png"
        cv2.imwrite(str(out_png), overlay)
        
        logger.info(f"[{image_id}] Generated {len(new_bridges)} candidate bridges to close gaps.")
        return new_bridges
