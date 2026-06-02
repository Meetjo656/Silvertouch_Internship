from pathlib import Path
from shapely.geometry import Polygon
from shapely.ops import unary_union
from src.utils.logger import get_logger

logger = get_logger(__name__)

class StructuralConnectivityValidator:
    """Stage 5: Structural Connectivity Validation. Removes floating isolated polygons."""
    
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def validate_connectivity(self, image_id: str, faces: list) -> list:
        logger.info(f"[{image_id}] Running Structural Connectivity Validation...")
        
        if not faces: return []
        
        # Determine the outer blueprint perimeter
        all_poly = unary_union(faces)
        outer_perimeter = all_poly.exterior if all_poly.geom_type == 'Polygon' else None
        
        if not outer_perimeter and all_poly.geom_type == 'MultiPolygon':
            # Use the exterior of the largest polygon as a proxy for the main perimeter
            largest = max(all_poly.geoms, key=lambda p: p.area)
            outer_perimeter = largest.exterior
            
        valid_faces = []
        removed = 0
        
        for i, face in enumerate(faces):
            touches_perimeter = False
            if outer_perimeter and face.exterior.intersects(outer_perimeter):
                touches_perimeter = True
                
            touches_neighbor = False
            for j, other_face in enumerate(faces):
                if i == j: continue
                if face.touches(other_face):
                    touches_neighbor = True
                    break
                    
            if touches_perimeter or touches_neighbor:
                valid_faces.append(face)
            else:
                removed += 1
                
        logger.info(f"[{image_id}] Removed {removed} isolated floating faces.")
        return valid_faces
