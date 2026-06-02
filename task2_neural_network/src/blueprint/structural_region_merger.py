import math
from pathlib import Path
from shapely.geometry import Polygon
from shapely.ops import unary_union
from src.utils.logger import get_logger

logger = get_logger(__name__)

class StructuralRegionMerger:
    """Stage 4: Structural Region Merging. Merges adjacent faces that belong to the same region."""
    
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def _is_orientation_consistent(self, f1: Polygon, f2: Polygon, merged: Polygon) -> bool:
        """Heuristic: if the merged polygon doesn't drastically decrease compactness, it's consistent."""
        c1 = (4 * math.pi * f1.area) / (f1.length ** 2) if f1.length > 0 else 0
        c2 = (4 * math.pi * f2.area) / (f2.length ** 2) if f2.length > 0 else 0
        c_merged = (4 * math.pi * merged.area) / (merged.length ** 2) if merged.length > 0 else 0
        
        # Consistent if merged compactness isn't terribly worse than the average of the two
        # or if the merged shape is reasonably convex.
        convexity = merged.area / merged.convex_hull.area if merged.convex_hull.area > 0 else 0
        
        return convexity > 0.6 # A very relaxed threshold for typical carton shapes
        
    def merge_regions(self, image_id: str, faces: list) -> list:
        logger.info(f"[{image_id}] Running Structural Region Merger...")
        
        if not faces: return []
        
        merged_faces = list(faces)
        merges_performed = 0
        changed = True
        
        while changed:
            changed = False
            n = len(merged_faces)
            i_merge, j_merge = -1, -1
            
            for i in range(n):
                for j in range(i + 1, n):
                    f1 = merged_faces[i]
                    f2 = merged_faces[j]
                    
                    if f1.touches(f2):
                        intersection = f1.intersection(f2)
                        shared_border = intersection.length
                        
                        min_peri = min(f1.length, f2.length)
                        
                        # 1. Shared border > 80% of smaller face perimeter
                        # Note: 80% perimeter means the face is almost entirely engulfed by the other face.
                        # This effectively removes deep 'bites' or almost-contained regions, while preventing
                        # valid adjacent panels (which share < 50% of their perimeter) from merging.
                        
                        condition1 = shared_border > 0.80 * min_peri
                        
                        if condition1:
                            merged_poly = unary_union([f1, f2])
                            
                            # 2. & 3. Valid polygon and orientation consistent
                            if merged_poly.geom_type == 'Polygon' and merged_poly.is_valid:
                                if self._is_orientation_consistent(f1, f2, merged_poly):
                                    i_merge, j_merge = i, j
                                    break
                if i_merge != -1:
                    break
                    
            if i_merge != -1:
                # Perform the merge
                f1 = merged_faces.pop(j_merge)
                f2 = merged_faces.pop(i_merge)
                new_poly = unary_union([f1, f2])
                merged_faces.append(new_poly)
                merges_performed += 1
                changed = True
                
        logger.info(f"[{image_id}] Performed {merges_performed} merge operations.")
        return merged_faces
