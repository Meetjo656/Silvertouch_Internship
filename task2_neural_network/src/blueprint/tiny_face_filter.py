import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TinyFaceFilter:
    """Stage 3: Tiny Face Removal. Removes microscopic faces representing text/noise."""
    
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def filter_tiny_faces(self, image_id: str, faces: list, original_img: np.ndarray) -> list:
        logger.info(f"[{image_id}] Running Tiny Face Filter...")
        
        if not faces: return []
        
        h, w = original_img.shape[:2]
        carton_area = h * w
        
        filtered_faces = []
        removed_count = 0
        
        for face in faces:
            if not isinstance(face, Polygon) or face.is_empty:
                continue
                
            relative_area = face.area / carton_area
            if relative_area >= 0.01:
                filtered_faces.append(face)
            else:
                removed_count += 1
                
        logger.info(f"[{image_id}] Removed {removed_count} tiny faces.")
        return filtered_faces
