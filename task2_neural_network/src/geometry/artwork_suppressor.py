import cv2
import numpy as np
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ArtworkSuppressor:
    """Stage 0: Suppresses high-frequency artwork noise (text, barcodes) while preserving structural boundaries."""
    
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def process(self, image_id: str, image: np.ndarray) -> np.ndarray:
        """Runs the artwork suppression pipeline."""
        logger.info(f"[{image_id}] Running Artwork Suppression...")
        
        # 1. Grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 3. Bilateral Filtering
        # Preserves edges but smooths out flat regions
        bilateral = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
        
        # 4. Adaptive Thresholding (to find high contrast elements like text and borders)
        thresh = cv2.adaptiveThreshold(
            bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
        )
        
        # 5. Connected Component Analysis (Filter small elements)
        h, w = thresh.shape
        total_area = h * w
        min_area = total_area * 0.0001 # Extremely small components
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        filtered_components = np.zeros_like(thresh)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                filtered_components[labels == i] = 255
                
        # 6. Morphological Opening (Remove remaining thin strokes / text)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(filtered_components, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Finally, we want the "suppressed" image to pass to Canny.
        # We can mask the bilateral image to soften the removed text areas.
        # Actually, the user pipeline says: Grayscale -> CLAHE -> Bilateral -> Adaptive Thresh -> Morph Closing -> Canny
        # Let's output the opened/filtered image as a structural mask to guide edge detection
        # Or, just return the filtered thresh map which acts as the clean edge precursor.
        
        out_suppressed = self.out_dir / f"{image_id}_artwork_suppressed.png"
        out_cc = self.out_dir / f"{image_id}_connected_components.png"
        
        cv2.imwrite(str(out_suppressed), bilateral)
        cv2.imwrite(str(out_cc), filtered_components)
        
        return bilateral, opened
