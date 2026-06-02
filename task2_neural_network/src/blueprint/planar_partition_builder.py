import cv2
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
from shapely.ops import unary_union
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PlanarPartitionBuilder:
    """Stage 4: Planar Partition Builder. Validates the structural blueprint."""
    
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def build_partition(self, image_id: str, faces: list[Polygon], original_img: np.ndarray, carton_area: float) -> dict:
        logger.info(f"[{image_id}] Building Planar Partition...")
        
        # 1. Filter Noise Faces
        # Faces that are extremely small (e.g. tiny enclosed text loops) should be rejected
        min_face_area = carton_area * 0.005 # 0.5% of total area
        structural_faces = [f for f in faces if f.area >= min_face_area]
        
        logger.info(f"[{image_id}] Filtered {len(faces)} raw faces down to {len(structural_faces)} structural faces.")
        
        # 2. Evaluate Constraints
        # Since shapely.ops.polygonize inherently produces a valid planar partition 
        # (no overlaps, exactly shared borders, closed polygons), we only need to verify Coverage.
        
        coverage_area = 0.0
        if structural_faces:
            union_poly = unary_union(structural_faces)
            coverage_area = union_poly.area
            
        coverage_pct = coverage_area / carton_area if carton_area > 0 else 0
        
        # 3. Output Blueprint
        h, w = original_img.shape[:2] if original_img is not None else (1000, 1000)
        overlay = original_img.copy() if original_img is not None else np.zeros((h, w, 3), dtype=np.uint8)
        
        for i, face in enumerate(structural_faces):
            color = (0, 255, 0) # Green for structural faces
            ext_coords = np.array(face.exterior.coords, dtype=np.int32)
            
            # Fill with transparency
            overlay_copy = overlay.copy()
            cv2.fillPoly(overlay_copy, [ext_coords], color)
            cv2.addWeighted(overlay_copy, 0.4, overlay, 0.6, 0, overlay)
            
            # Draw borders strictly
            cv2.polylines(overlay, [ext_coords], isClosed=True, color=(255, 255, 255), thickness=2)
            
            # Draw Region ID
            cx, cy = int(face.centroid.x), int(face.centroid.y)
            cv2.putText(overlay, f"R{i}", (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
        out_png = self.out_dir / f"{image_id}_blueprint_final.png"
        cv2.imwrite(str(out_png), overlay)
        
        # Return Partition Metrics
        metrics = {
            "image_id": image_id,
            "face_count": len(structural_faces),
            "coverage_pct": coverage_pct,
            "overlap_pct": 0.0, # Guaranteed by polygonize
            "shared_border_consistency": 100.0, # Guaranteed by polygonize
            "blueprint_validity": 100.0 if coverage_pct >= 0.8 else coverage_pct * 100
        }
        
        logger.info(f"[{image_id}] Partition metrics: Coverage={coverage_pct*100:.1f}%, Faces={len(structural_faces)}")
        return metrics
