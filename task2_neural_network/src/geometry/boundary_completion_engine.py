import cv2
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
from src.utils.logger import get_logger

from src.geometry.perimeter_hypothesis_builder import PerimeterHypothesisBuilder
from src.geometry.silhouette_fusion import SilhouetteFusion
from src.geometry.gap_bridge_solver import GapBridgeSolver

logger = get_logger(__name__)

class BoundaryCompletionEngine:
    """Stage 6.4: Orchestrates the Boundary Completion Phase."""
    
    def __init__(self, project_root: Path, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.perimeter_builder = PerimeterHypothesisBuilder(out_dir)
        self.silhouette_fusion = SilhouetteFusion(project_root, out_dir)
        self.gap_solver = GapBridgeSolver(out_dir)
        
    def complete_boundary(self, image_id: str, img_meta, pslg_edges: list, faces: list, original_img: np.ndarray) -> list:
        logger.info(f"[{image_id}] Running Boundary Completion Engine...")
        
        # 1. Perimeter Hypothesis
        hypothesis_hull = self.perimeter_builder.build_hypothesis(image_id, faces, pslg_edges, original_img)
        
        # 2. Silhouette Fusion (SAM as weak geometric fence)
        silhouette_poly = self.silhouette_fusion.extract_silhouette(image_id, img_meta, original_img)
        
        # 3. Gap Bridge Solver
        # Uses silhouette if available, otherwise bridges to endpoints
        new_bridges = self.gap_solver.solve_gaps(image_id, pslg_edges, silhouette_poly, original_img)
        
        # 4. Completed Boundary Network
        completed_network = list(pslg_edges) + new_bridges
        if silhouette_poly is not None and not silhouette_poly.is_empty:
            if isinstance(silhouette_poly, Polygon):
                completed_network.append(silhouette_poly.exterior)
            else:
                for geom in silhouette_poly.geoms:
                    completed_network.append(geom.exterior)
        
        # Draw overlay
        h, w = original_img.shape[:2] if original_img is not None else (1000, 1000)
        overlay = original_img.copy() if original_img is not None else np.zeros((h, w, 3), dtype=np.uint8)
        
        for edge in completed_network:
            ext_coords = np.array(edge.coords, dtype=np.int32)
            cv2.polylines(overlay, [ext_coords], isClosed=False, color=(0, 255, 0), thickness=2)
            
        out_png = self.out_dir / f"{image_id}_completed_boundary.png"
        cv2.imwrite(str(out_png), overlay)
        
        logger.info(f"[{image_id}] Boundary Completion added {len(new_bridges)} bridges. Network size: {len(completed_network)}.")
        return completed_network
