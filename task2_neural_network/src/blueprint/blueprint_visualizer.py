import cv2
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.blueprint.fragment_merger import MergedRegion
from src.utils.logger import get_logger

logger = get_logger(__name__)

class BlueprintVisualizer:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.out_dir = self.project_root / "data" / "interim" / "blueprints"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.class_colors = {
            "PANEL": (0, 255, 0),     # Green
            "FLAP": (255, 165, 0),    # Orange
            "GLUE_TAB": (0, 0, 255),  # Red
            "DUST_FLAP": (255, 0, 255),# Magenta
            "BACKGROUND": (128, 128, 128) # Gray
        }
        
    def _draw_polygon(self, img: np.ndarray, poly: Polygon, color: tuple, thickness: int = 2):
        if poly.is_empty: return
        
        # Draw exterior
        ext_coords = np.array(poly.exterior.coords, dtype=np.int32)
        cv2.polylines(img, [ext_coords], isClosed=True, color=color, thickness=thickness)
        
        # Draw interiors (holes)
        for interior in poly.interiors:
            int_coords = np.array(interior.coords, dtype=np.int32)
            cv2.polylines(img, [int_coords], isClosed=True, color=(0, 255, 255), thickness=1) # Yellow holes
            
    def render_blueprint(self, image_id: str, image: np.ndarray, merged_regions: list[MergedRegion]):
        """Renders the final merged regions on top of the original image."""
        display = image.copy()
        overlay = image.copy()
        
        for region in merged_regions:
            color = self.class_colors.get(region.structural_class, (255, 255, 255))
            
            # Fill Polygon on overlay
            if not region.polygon.is_empty:
                # Convert exterior to integer points for fillPoly
                ext_pts = np.array(region.polygon.exterior.coords, dtype=np.int32)
                cv2.fillPoly(overlay, [ext_pts], color)
                
                # Mask out holes
                for interior in region.polygon.interiors:
                    int_pts = np.array(interior.coords, dtype=np.int32)
                    cv2.fillPoly(overlay, [int_pts], (0, 0, 0)) # Fill holes with black to simulate void
                    
            # Draw boundary on display
            self._draw_polygon(display, region.polygon, color, thickness=3)
            
            # Add text label (Region ID and Class) at centroid
            centroid = region.polygon.centroid
            if not centroid.is_empty:
                cx, cy = int(centroid.x), int(centroid.y)
                text = f"{region.structural_class}\n{region.region_id}"
                
                # Basic multi-line putText
                y_offset = cy
                for line in text.split('\n'):
                    cv2.putText(display, line, (cx - 40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display, line, (cx - 40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    y_offset += 25
                    
        # Blend overlay
        cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
        
        out_path = self.out_dir / f"{image_id}_blueprint.jpg"
        cv2.imwrite(str(out_path), display)
        logger.info(f"Saved blueprint visualization: {out_path}")
