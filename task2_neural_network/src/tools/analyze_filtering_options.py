import json
import cv2
import pandas as pd
from pathlib import Path
from src.data.metadata_mgr import MetadataManager
from src.features.geometric import extract_geometric_features

def analyze_filtering():
    project_root = Path(".")
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    mgr = MetadataManager(metadata_path)
    
    data = []
    
    for img_meta in mgr.data:
        image_id = img_meta.image_id
        image_path = project_root / img_meta.image_path
        image = cv2.imread(str(image_path))
        if image is None: continue
        
        img_area = image.shape[0] * image.shape[1]
        
        # Load masks and calculate area
        masks_list = []
        for mask_meta in img_meta.masks:
            mask_path = project_root / mask_meta.mask_path
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                area = cv2.countNonZero(mask)
                masks_list.append({
                    "mask_id": mask_meta.mask_id,
                    "mask": mask,
                    "area": area
                })
        
        # Sort by area
        masks_list.sort(key=lambda x: x["area"], reverse=True)
        top_20 = masks_list[:20]
        
        for rank, m in enumerate(top_20):
            # Extract geometric features
            features = extract_geometric_features(m["mask"], image_area=img_area)
            features["image_id"] = image_id
            features["mask_id"] = m["mask_id"]
            features["rank"] = rank + 1
            data.append(features)
            
    df = pd.DataFrame(data)
    
    # 1. Analyze Area Rank Filter + Rel Area + Geometric Quality
    # A candidate is good if: solidity > 0.5 and extent > 0.35 and relative_area > 0.01
    df["passes_geom"] = (df["solidity"] >= 0.5) & (df["extent"] >= 0.4) & (df["circularity"] >= 0.2)
    df["passes_rel_area"] = df["relative_area"] >= 0.015
    
    df["is_structural_candidate"] = df["passes_geom"] & df["passes_rel_area"]
    
    print("=== Filter Simulation Results ===")
    
    for n in [5, 8, 10]:
        df_top_n = df[df["rank"] <= n]
        total_masks = len(df_top_n)
        structural_candidates = df_top_n[df_top_n["is_structural_candidate"]]
        sc_count = len(structural_candidates)
        avg_per_image = sc_count / 7.0
        print(f"Option: Top {n}")
        print(f"  Total Candidates BEFORE geom/area filter: {total_masks}")
        print(f"  Total Candidates AFTER geom/area filter: {sc_count}")
        print(f"  Expected valid masks per image: {avg_per_image:.1f}")
        
    df.to_csv("data/interim/filtering_analysis.csv", index=False)

if __name__ == "__main__":
    analyze_filtering()
