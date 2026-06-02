from pathlib import Path
from shapely.geometry import Polygon
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ContainmentFilter:
    """Stage 2: Containment Removal. Removes faces completely contained within another face's exterior."""
    
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def filter_containment(self, image_id: str, faces: list) -> list:
        logger.info(f"[{image_id}] Running Containment Filter...")
        
        if not faces: return []
        
        to_remove = set()
        n = len(faces)
        
        for i in range(n):
            if i in to_remove: continue
            poly_i = faces[i]
            if not isinstance(poly_i, Polygon) or poly_i.is_empty:
                continue
                
            for j in range(n):
                if i == j or j in to_remove: continue
                poly_j = faces[j]
                if not isinstance(poly_j, Polygon) or poly_j.is_empty:
                    continue
                    
                # Check if poly_i is contained within the *exterior* of poly_j
                # Because polygonize creates non-overlapping faces, poly_j might have a hole where poly_i is.
                # So we test against the solid exterior polygon.
                solid_j = Polygon(poly_j.exterior)
                
                if solid_j.contains(poly_i):
                    to_remove.add(i)
                    break # i is removed, no need to check further j's
                    
        filtered_faces = [f for i, f in enumerate(faces) if i not in to_remove]
        
        logger.info(f"[{image_id}] Removed {len(to_remove)} contained faces.")
        return filtered_faces
