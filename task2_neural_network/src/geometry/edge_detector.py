import cv2
import numpy as np
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

class EdgeDetector:
    """Stage 1: Structural Edge Recovery."""
    
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def detect_edges(self, image_id: str, suppressed_img: np.ndarray, cleaned_binary: np.ndarray) -> np.ndarray:
        """Runs Canny edge detection. We use the cleaned binary as a strong edge guide."""
        logger.info(f"[{image_id}] Running Structural Edge Recovery...")
        
        # 1. Morphological Closing (Connect broken structural lines)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(cleaned_binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 2. Canny Edge Detection
        # Use the closed binary image to find crisp edges instead of the noisy bilateral image
        edges = cv2.Canny(closed, 50, 150, apertureSize=3)
        
        out_path = self.out_dir / f"{image_id}_edge_map.png"
        cv2.imwrite(str(out_path), edges)
        
        return edges
