import cv2
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon, MultiLineString, LineString, MultiPolygon
from shapely.ops import unary_union
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PerimeterHypothesisBuilder:
    """Stage 6.1: Generates a candidate outer boundary from recovered faces and regularized lines."""
    
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def build_hypothesis(self, image_id: str, recovered_faces: list, pslg_edges: list, original_img: np.ndarray):
        logger.info(f"[{image_id}] Building Perimeter Hypothesis...")
        
        # 1. Union existing valid faces (if any)
        known_area_poly = None
        if recovered_faces:
            known_area_poly = unary_union(recovered_faces)
            
        # 2. Extract Exterior of the Line Graph
        # A simple perimeter hypothesis is the convex hull of all lines
        all_points = []
        for edge in pslg_edges:
            all_points.extend(list(edge.coords))
            
        hypothesis_hull = None
        if all_points:
            from shapely.geometry import MultiPoint
            mp = MultiPoint(all_points)
            hypothesis_hull = mp.convex_hull
            
        # Draw overlay
        h, w = original_img.shape[:2] if original_img is not None else (1000, 1000)
        overlay = original_img.copy() if original_img is not None else np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw Known Faces
        if known_area_poly:
            if isinstance(known_area_poly, Polygon):
                polys = [known_area_poly]
            else:
                polys = list(known_area_poly.geoms)
                
            for poly in polys:
                ext_coords = np.array(poly.exterior.coords, dtype=np.int32)
                cv2.polylines(overlay, [ext_coords], isClosed=True, color=(0, 255, 0), thickness=2)
                
        # Draw Convex Hull Hypothesis
        if hypothesis_hull and isinstance(hypothesis_hull, Polygon):
            hull_coords = np.array(hypothesis_hull.exterior.coords, dtype=np.int32)
            cv2.polylines(overlay, [hull_coords], isClosed=True, color=(255, 0, 255), thickness=3)
            
        out_png = self.out_dir / f"{image_id}_perimeter_hypothesis.png"
        cv2.imwrite(str(out_png), overlay)
        
        return hypothesis_hull
