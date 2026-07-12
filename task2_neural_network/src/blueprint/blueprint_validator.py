import numpy as np
import cv2
from pathlib import Path
from shapely.geometry import Polygon
from shapely.ops import unary_union
from typing import List, Dict

from src.blueprint.fragment_merger import MergedRegion
from src.utils.logger import get_logger

logger = get_logger(__name__)

class BlueprintValidator:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        
    def evaluate(self, all_merged_regions: Dict[str, List[MergedRegion]], mgr_data):
        """Evaluates the merged regions and computes validity scores per image and overall."""
        results = []
        overall_metrics = {
            "total_coverage": [],
            "total_overlap": [],
            "total_regions": [],
            "validity_scores": []
        }
        
        for img_meta in mgr_data:
            image_id = img_meta.image_id
            regions = all_merged_regions.get(image_id, [])
            
            # Load image for bounds
            img_path = self.project_root / img_meta.image_path
            image = cv2.imread(str(img_path))
            if image is None: continue
            
            h, w = image.shape[:2]
            carton_area = h * w # Approximation: Total image area
            
            if not regions:
                results.append({
                    "image_id": image_id,
                    "coverage": 0,
                    "overlap": 0,
                    "region_count": 0,
                    "score": 0
                })
                continue
                
            polygons = [r.polygon for r in regions]
            
            # Coverage: Area of the union of all regions
            try:
                union_poly = unary_union(polygons)
                coverage_area = union_poly.area
            except:
                coverage_area = sum(p.area for p in polygons) # Fallback if invalid
                
            coverage_pct = coverage_area / carton_area if carton_area > 0 else 0
            
            # Overlap: Sum of intersection areas between all pairs of merged regions
            total_overlap_area = 0.0
            n = len(polygons)
            for i in range(n):
                for j in range(i + 1, n):
                    if polygons[i].intersects(polygons[j]):
                        try:
                            inter = polygons[i].intersection(polygons[j])
                            total_overlap_area += inter.area
                        except:
                            pass
                            
            overlap_pct = total_overlap_area / carton_area if carton_area > 0 else 0
            
            # Region Count
            region_count = len(regions)
            
            # Blueprint Validity Score
            # Perfect score = 100
            # Penalty for low coverage (ideally > 50% for carton)
            # Heavy penalty for overlap
            # Penalty for excessive fragmentation (> 10 regions)
            
            coverage_score = min(coverage_pct * 100, 100) # 0 to 100
            overlap_penalty = overlap_pct * 500 # 1% overlap = 5 point penalty
            fragment_penalty = max(0, region_count - 10) * 5 # 5 points per region over 10
            
            validity_score = coverage_score - overlap_penalty - fragment_penalty
            validity_score = max(0, min(100, validity_score))
            
            results.append({
                "image_id": image_id,
                "coverage": coverage_pct,
                "overlap": overlap_pct,
                "region_count": region_count,
                "score": validity_score
            })
            
            overall_metrics["total_coverage"].append(coverage_pct)
            overall_metrics["total_overlap"].append(overlap_pct)
            overall_metrics["total_regions"].append(region_count)
            overall_metrics["validity_scores"].append(validity_score)
            
        return results, overall_metrics
