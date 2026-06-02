import cv2
import numpy as np
from pathlib import Path

from src.data.metadata_mgr import MetadataManager
from src.features.geometric import extract_geometric_features
from src.core.labels import StructuralClass

def auto_label_multiclass_candidates(num_to_label=20):
    project_root = Path(".")
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    mgr = MetadataManager(metadata_path)
    
    labeled_count = 0
    counts = {
        StructuralClass.PANEL.value: 0,
        StructuralClass.FLAP.value: 0,
        StructuralClass.GLUE_TAB.value: 0,
        StructuralClass.DUST_FLAP.value: 0,
        StructuralClass.BACKGROUND.value: 0,
        StructuralClass.UNKNOWN.value: 0 # Ambiguous
    }
    
    notes = []
    
    for img_meta in mgr.data:
        image_id = img_meta.image_id
        img_w, img_h = img_meta.resolution
        img_area = img_w * img_h
        
        for mask_meta in img_meta.masks:
            if labeled_count >= num_to_label:
                break
                
            if not getattr(mask_meta, "is_candidate", True):
                continue
                
            if mask_meta.structural_class != StructuralClass.UNKNOWN.value:
                continue
                
            mask_path = project_root / mask_meta.mask_path
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if mask_img is None:
                continue
                
            features = extract_geometric_features(mask_img, image_area=img_area)
            
            # Multiclass Heuristic
            # Large, highly rectangular masks -> PANEL
            if features["relative_area"] > 0.05 and features["solidity"] > 0.8:
                label = StructuralClass.PANEL.value
            # Medium, rectangular masks -> FLAP
            elif 0.01 < features["relative_area"] <= 0.05 and features["solidity"] > 0.8:
                label = StructuralClass.FLAP.value
            # Small, very rectangular and narrow -> GLUE_TAB
            elif 0.002 < features["relative_area"] <= 0.01 and features["solidity"] > 0.8 and features["aspect_ratio"] > 3.0:
                label = StructuralClass.GLUE_TAB.value
            # Small, rectangular, square-ish -> DUST_FLAP
            elif 0.002 < features["relative_area"] <= 0.01 and features["solidity"] > 0.8 and features["aspect_ratio"] <= 3.0:
                label = StructuralClass.DUST_FLAP.value
            # High complexity -> BACKGROUND (Text/noise)
            elif features["vertex_count"] > 15 or features["solidity"] < 0.6:
                label = StructuralClass.BACKGROUND.value
            # Anything else -> UNKNOWN / Ambiguous
            else:
                label = StructuralClass.UNKNOWN.value
                notes.append(f"Ambiguity noted on {mask_meta.mask_id}: Relative Area {features['relative_area']:.4f}, Solidity {features['solidity']:.4f}")
                
            mgr.update_mask_label(image_id, mask_meta.mask_id, label, "auto_multiclass_expert")
            counts[label] += 1
            labeled_count += 1
            
        if labeled_count >= num_to_label:
            break
            
    print(f"\n--- Multiclass Labeling Session Complete ({labeled_count} masks) ---")
    print(f"Number of Panels: {counts[StructuralClass.PANEL.value]}")
    print(f"Number of Flaps: {counts[StructuralClass.FLAP.value]}")
    print(f"Number of Glue Tabs: {counts[StructuralClass.GLUE_TAB.value]}")
    print(f"Number of Dust Flaps: {counts[StructuralClass.DUST_FLAP.value]}")
    print(f"Number of Ambiguous Masks: {counts[StructuralClass.UNKNOWN.value]}")
    print(f"Number of Background Masks: {counts[StructuralClass.BACKGROUND.value]}")
    
    print("\n--- Notes on Ambiguity ---")
    for note in notes:
        print(note)
    if not notes:
        print("No ambiguity encountered.")

if __name__ == "__main__":
    auto_label_multiclass_candidates(20)
