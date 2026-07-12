import math
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FaceCleanupAudit:
    """Stage 1: Face Audit. Analyzes geometric properties of extracted faces."""
    
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def audit_faces(self, image_id: str, faces: list, original_img: np.ndarray) -> dict:
        logger.info(f"[{image_id}] Running Face Audit...")
        
        h, w = original_img.shape[:2]
        carton_area = h * w
        
        metrics = []
        n = len(faces)
        
        for i, face in enumerate(faces):
            if not isinstance(face, Polygon) or face.is_empty:
                continue
                
            area = face.area
            relative_area = area / carton_area
            perimeter = face.length
            
            compactness = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            
            hull = face.convex_hull
            convexity = area / hull.area if hull.area > 0 else 0
            
            # Find neighbors and shared border lengths
            neighbors = 0
            shared_lengths = []
            for j, other_face in enumerate(faces):
                if i == j: continue
                if face.touches(other_face):
                    intersection = face.intersection(other_face)
                    if intersection.geom_type in ['LineString', 'MultiLineString']:
                        neighbors += 1
                        shared_lengths.append(intersection.length)
                        
            # Check containment (is this face contained in any other face?)
            contained_in = []
            contains = []
            for j, other_face in enumerate(faces):
                if i == j: continue
                # We use contains to check if other_face completely contains face
                # Since polygons from polygonize shouldn't overlap, full containment is rare
                # unless there are holes or un-polygonized areas. But if it happens:
                if other_face.contains(face):
                    contained_in.append(j)
                if face.contains(other_face):
                    contains.append(j)
                    
            metrics.append({
                "face_id": i,
                "area": area,
                "relative_area": relative_area,
                "perimeter": perimeter,
                "compactness": compactness,
                "convexity": convexity,
                "neighbors": neighbors,
                "shared_lengths": shared_lengths,
                "contained_in": contained_in,
                "contains": contains
            })
            
        return {"metrics": metrics, "total_faces": n}
