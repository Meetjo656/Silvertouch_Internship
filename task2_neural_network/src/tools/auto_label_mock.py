import json
import cv2
import numpy as np
from pathlib import Path

from src.data.metadata_mgr import MetadataManager
from src.features.geometric import extract_geometric_features

def auto_label_candidates():
    project_root = Path(".")
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    mgr = MetadataManager(metadata_path)
    
    labeled_count = 0
    structural_count = 0
    non_structural_count = 0
    
    for img_meta in mgr.data:
        image_id = img_meta.image_id
        img_w, img_h = img_meta.resolution
        img_area = img_w * img_h
        
        for mask_meta in img_meta.masks:
            if not getattr(mask_meta, "is_candidate", True):
                continue
                
            mask_path = project_root / mask_meta.mask_path
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if mask_img is None:
                continue
                
            # Use our exact feature extraction logic to make a heuristic guess
            features = extract_geometric_features(mask_img, image_area=img_area)
            
            # Simple Heuristic for Mock Labeling
            # 1. Very large masks are panels
            # 2. Smooth masks with few vertices are flaps/tabs
            # 3. Highly complex, jagged masks with many vertices are text/noise
            
            if features["relative_area"] > 0.05:
                label = "structural"
            elif features["vertex_count"] > 0 and features["vertex_count"] < 15 and features["solidity"] > 0.8:
                label = "structural"
            else:
                label = "non_structural"
                
            mgr.update_mask_label(image_id, mask_meta.mask_id, label, "auto_heuristic")
            labeled_count += 1
            if label == "structural":
                structural_count += 1
            else:
                non_structural_count += 1
                
    print(f"Auto-labeled {labeled_count} candidate masks.")
    print(f"Structural: {structural_count}")
    print(f"Non-Structural: {non_structural_count}")

if __name__ == "__main__":
    auto_label_candidates()
