import math
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

class LineDensityAnalysis:
    """Task 5: Local Line Density Analysis."""
    
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_midpoint(self, line):
        return ((line["x1"] + line["x2"]) / 2, (line["y1"] + line["y2"]) / 2)
        
    def analyze_density(self, image_id: str, lines: list, original_img) -> dict:
        """
        Computes local line density (neighbor count within radius) for each line.
        """
        logger.info(f"[{image_id}] Running Line Density Analysis...")
        
        h, w = original_img.shape[:2]
        diag = math.hypot(h, w)
        radius = 0.05 * diag
        
        midpoints = [self._get_midpoint(l) for l in lines]
        n = len(lines)
        densities = [0] * n
        
        for i in range(n):
            count = 0
            mx1, my1 = midpoints[i]
            for j in range(n):
                if i == j: continue
                mx2, my2 = midpoints[j]
                
                dist = math.hypot(mx1 - mx2, my1 - my2)
                if dist <= radius:
                    count += 1
            densities[i] = count
            
        return {
            "densities": densities,
            "radius": radius
        }
