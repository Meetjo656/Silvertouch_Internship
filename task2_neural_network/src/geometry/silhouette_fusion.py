import cv2
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SilhouetteFusion:
    """Stage 6.2: Reintroduces SAM ONLY as weak geometric evidence to form a boundary fence."""
    
    def __init__(self, project_root: Path, out_dir: Path):
        self.project_root = project_root
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_silhouette(self, image_id: str, img_meta, original_img: np.ndarray) -> Polygon:
        logger.info(f"[{image_id}] Running Silhouette Fusion...")
        
        h, w = original_img.shape[:2]
        fused_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Iterate over all SAM candidates and union them to form the outer silhouette
        valid_masks_found = 0
        for mask_obj in img_meta.masks:
            mask_path = self.project_root / mask_obj.mask_path
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                # Binarize
                _, mask_img = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
                fused_mask = cv2.bitwise_or(fused_mask, mask_img)
                valid_masks_found += 1
                
        logger.info(f"[{image_id}] Fused {valid_masks_found} SAM masks into silhouette.")
        
        # Find the outer contour of the fused mask
        contours, _ = cv2.findContours(fused_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        silhouette_poly = None
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # Simplify the contour slightly to remove SAM jaggy pixelation
            epsilon = 0.005 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            if len(approx) >= 3:
                points = [tuple(point[0]) for point in approx]
                try:
                    silhouette_poly = Polygon(points)
                    if not silhouette_poly.is_valid:
                        silhouette_poly = silhouette_poly.buffer(0)
                except Exception as e:
                    logger.error(f"[{image_id}] Failed to create Silhouette Polygon: {e}")
                    
        # Draw overlay
        overlay = original_img.copy() if original_img is not None else np.zeros((h, w, 3), dtype=np.uint8)
        
        if silhouette_poly and isinstance(silhouette_poly, Polygon):
            ext_coords = np.array(silhouette_poly.exterior.coords, dtype=np.int32)
            cv2.polylines(overlay, [ext_coords], isClosed=True, color=(0, 255, 255), thickness=4)
            # Draw semi-transparent fill
            overlay_copy = overlay.copy()
            cv2.fillPoly(overlay_copy, [ext_coords], (0, 255, 255))
            cv2.addWeighted(overlay_copy, 0.2, overlay, 0.8, 0, overlay)
            
        out_png = self.out_dir / f"{image_id}_silhouette_overlay.png"
        cv2.imwrite(str(out_png), overlay)
        
        return silhouette_poly
