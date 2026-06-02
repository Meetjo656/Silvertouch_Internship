import cv2
import numpy as np
import json
from pathlib import Path
from shapely.geometry import LineString, Polygon, mapping
from shapely.ops import polygonize
from src.utils.logger import get_logger

logger = get_logger(__name__)

class Polygonizer:
    """Stage 3: Polygonization. Extracts closed faces from CAD-like line segments."""
    
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def polygonize_lines(self, image_id: str, regularized_lines: list, original_img: np.ndarray) -> list:
        logger.info(f"[{image_id}] Running Polygonization...")
        
        # Convert dictionaries to Shapely LineStrings if they are dicts
        shapely_lines = []
        for line in regularized_lines:
            if isinstance(line, dict):
                p1 = (line["x1"], line["y1"])
                p2 = (line["x2"], line["y2"])
                shapely_lines.append(LineString([p1, p2]))
            else:
                shapely_lines.append(line)
                
        # The magic function: extracts all closed, non-overlapping planar polygons
        polygons = list(polygonize(shapely_lines))
        
        logger.info(f"[{image_id}] Extracted {len(polygons)} closed faces.")
        
        # Draw overlay
        h, w = original_img.shape[:2] if original_img is not None else (1000, 1000)
        overlay = original_img.copy() if original_img is not None else np.zeros((h, w, 3), dtype=np.uint8)
        
        geojson_features = []
        
        for i, poly in enumerate(polygons):
            # Draw random colored filled polygon
            color = tuple(np.random.randint(50, 255, 3).tolist())
            ext_coords = np.array(poly.exterior.coords, dtype=np.int32)
            cv2.fillPoly(overlay, [ext_coords], color)
            cv2.polylines(overlay, [ext_coords], isClosed=True, color=(255, 255, 255), thickness=2)
            
            # Export GeoJSON format
            feature = {
                "type": "Feature",
                "id": f"face_{i}",
                "geometry": mapping(poly),
                "properties": {"area": poly.area}
            }
            geojson_features.append(feature)
            
        # Blend overlay
        if original_img is not None:
            cv2.addWeighted(overlay, 0.6, original_img, 0.4, 0, overlay)
            
        out_png = self.out_dir / f"{image_id}_polygon_faces.png"
        cv2.imwrite(str(out_png), overlay)
        
        geojson = {
            "type": "FeatureCollection",
            "features": geojson_features
        }
        
        out_json = self.out_dir / f"{image_id}_polygon_faces.geojson"
        with open(out_json, "w") as f:
            json.dump(geojson, f, indent=4)
            
        return polygons
