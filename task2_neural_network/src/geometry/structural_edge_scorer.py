import math
import numpy as np
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

class StructuralEdgeScorer:
    """Task 6: Structural Edge Scoring."""
    
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def _calculate_angle(self, line):
        dx = line["x2"] - line["x1"]
        dy = line["y2"] - line["y1"]
        angle = math.degrees(math.atan2(dy, dx)) % 180
        return angle
        
    def score_edges(self, image_id: str, lines: list, original_img: np.ndarray, analysis_results: dict) -> list:
        logger.info(f"[{image_id}] Scoring Structural Edges...")
        
        h, w = original_img.shape[:2]
        diag = math.hypot(h, w)
        
        # Unpack results
        persistence = analysis_results.get("persistence", {}).get("scores", [])
        conn = analysis_results.get("connectivity", {})
        comp_sizes = conn.get("component_sizes", [])
        perimeter = analysis_results.get("perimeter", {})
        affinity = perimeter.get("classifications", [])
        densities = analysis_results.get("density", {}).get("densities", [])
        
        max_comp_size = max(comp_sizes) if comp_sizes else 1
        max_density = max(densities) if densities else 1
        if max_density == 0: max_density = 1
        
        scores = []
        for i, line in enumerate(lines):
            # 1. Length Score
            length = math.hypot(line["x2"] - line["x1"], line["y2"] - line["y1"])
            # Normalize against an expected structural line length (e.g., 50% of diag)
            len_score = min((length / (0.5 * diag)) * 100, 100)
            
            # 2. Orientation Score
            angle = self._calculate_angle(line)
            # Orthogonal (0, 90, 180) -> 100, Diagonal -> 50, Irregular -> 0
            # Simple heuristic
            angle_dist_ortho = min(angle % 90, 90 - (angle % 90))
            if angle_dist_ortho <= 10:
                ori_score = 100
            elif 35 <= (angle % 90) <= 55:
                ori_score = 50
            else:
                ori_score = 0
                
            # 3. Connectivity Score
            conn_score = (comp_sizes[i] / max_comp_size) * 100
            
            # 4. Perimeter Affinity Score
            aff_score = 100 if affinity[i] == "PERIMETER_CONNECTED" else 0
            
            # 5. Persistence Score (1 to 4)
            pers_val = persistence[i] if i < len(persistence) else 1
            pers_score = (pers_val / 4.0) * 100
            
            # 6. Density Score (Inversely proportional)
            # High density = artwork = low score
            den_val = densities[i] if i < len(densities) else 0
            den_score = max(0, 100 - (den_val / max_density) * 100)
            
            # Final Score Calculation
            final_score = (
                0.15 * len_score +
                0.10 * ori_score +
                0.15 * conn_score +
                0.15 * aff_score +
                0.25 * pers_score +
                0.20 * den_score
            )
            
            scores.append({
                "line": line,
                "length_score": len_score,
                "orientation_score": ori_score,
                "connectivity_score": conn_score,
                "affinity_score": aff_score,
                "persistence_score": pers_score,
                "density_score": den_score,
                "final_score": final_score
            })
            
        return scores
