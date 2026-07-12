import cv2
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon

class FaceBoundaryRenderer:
    """Renders blueprint faces."""
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def render_clean_blueprint(self, image_id: str, faces: list, original_img: np.ndarray):
        h, w = original_img.shape[:2]
        blank_img = np.zeros((h, w, 3), dtype=np.uint8)
        overlay_img = original_img.copy()
        
        for i, face in enumerate(faces):
            if not isinstance(face, Polygon) or face.is_empty: continue
            
            # Extract exterior coordinates
            ext_coords = np.array(face.exterior.coords, dtype=np.int32)
            
            # Draw boundary outline
            cv2.polylines(blank_img, [ext_coords], True, (255, 255, 255), 2)
            cv2.polylines(overlay_img, [ext_coords], True, (0, 255, 0), 3)
            
            # Compute centroid for label
            cx, cy = int(face.centroid.x), int(face.centroid.y)
            label = f"R{i} (A:{int(face.area)})"
            
            # Draw Region ID and Area
            cv2.putText(blank_img, label, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(overlay_img, label, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
        cv2.imwrite(str(self.out_dir / f"{image_id}_clean_blueprint_boundaries.png"), blank_img)
        cv2.imwrite(str(self.out_dir / f"{image_id}_clean_blueprint_overlay.png"), overlay_img)
