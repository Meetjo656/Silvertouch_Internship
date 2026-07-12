import cv2
import numpy as np
import json
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

class LineDetector:
    """Stage 2: Structural Line Detection."""
    
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def detect_lines(self, image_id: str, edge_map: np.ndarray, original_img: np.ndarray) -> list:
        """Runs Probabilistic Hough Transform and filters tiny segments."""
        logger.info(f"[{image_id}] Running Structural Line Detection...")
        
        h, w = edge_map.shape
        diagonal = np.sqrt(h**2 + w**2)
        min_length = diagonal * 0.02
        
        # Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            edge_map,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=min_length,
            maxLineGap=10
        )
        
        filtered_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                filtered_lines.append({
                    "x1": int(x1), "y1": int(y1), 
                    "x2": int(x2), "y2": int(y2)
                })
                
        # Draw overlay
        overlay = original_img.copy() if original_img is not None else np.zeros((h, w, 3), dtype=np.uint8)
        for line in filtered_lines:
            cv2.line(overlay, (line["x1"], line["y1"]), (line["x2"], line["y2"]), (0, 0, 255), 2)
            
        out_png = self.out_dir / f"{image_id}_line_overlay.png"
        cv2.imwrite(str(out_png), overlay)
        
        out_json = self.out_dir / f"{image_id}_raw_lines.json"
        with open(out_json, "w") as f:
            json.dump({"lines": filtered_lines}, f, indent=4)
            
        logger.info(f"[{image_id}] Detected {len(filtered_lines)} raw structural lines.")
        return filtered_lines
