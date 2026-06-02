from pathlib import Path
import numpy as np
from src.geometry.multiscale_edge_analysis import MultiScaleEdgeAnalysis
from src.geometry.structural_connectivity import StructuralConnectivity
from src.geometry.perimeter_affinity import PerimeterAffinity
from src.geometry.line_density_analysis import LineDensityAnalysis
from src.geometry.structural_edge_scorer import StructuralEdgeScorer
from src.utils.logger import get_logger

logger = get_logger(__name__)

class StructuralEdgeClassifier:
    """Task 1: Structural Edge Analysis Framework.
    This module is ANALYSIS ONLY. It computes metrics but DOES NOT remove lines.
    """
    
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.multiscale = MultiScaleEdgeAnalysis(out_dir)
        self.connectivity = StructuralConnectivity(out_dir)
        self.affinity = PerimeterAffinity(out_dir)
        self.density = LineDensityAnalysis(out_dir)
        self.scorer = StructuralEdgeScorer(out_dir)
        
    def analyze_edges(self, image_id: str, regularized_lines: list, original_img: np.ndarray) -> list:
        logger.info(f"[{image_id}] Running Structural Edge Classifier (Analysis Phase)...")
        
        # 1. Multi-Scale Persistence
        pers_res = self.multiscale.analyze_persistence(image_id, original_img, regularized_lines)
        
        # 2. Connectivity Analysis
        conn_res = self.connectivity.analyze_connectivity(image_id, regularized_lines)
        
        # 3. Perimeter Affinity
        aff_res = self.affinity.analyze_affinity(image_id, regularized_lines, original_img)
        
        # 4. Line Density
        den_res = self.density.analyze_density(image_id, regularized_lines, original_img)
        
        analysis_results = {
            "persistence": pers_res,
            "connectivity": conn_res,
            "perimeter": aff_res,
            "density": den_res
        }
        
        # 5. Structural Edge Scoring
        scores = self.scorer.score_edges(image_id, regularized_lines, original_img, analysis_results)
        
        return scores
