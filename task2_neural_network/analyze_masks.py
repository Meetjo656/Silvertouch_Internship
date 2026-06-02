import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from src.data.metadata_mgr import MetadataManager

def analyze():
    project_root = Path(".")
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    interim_dir = project_root / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    mgr = MetadataManager(metadata_path)
    
    data = []
    
    for img_meta in mgr.data:
        image_id = img_meta.image_id
        # Use resolution directly from metadata instead of loading image
        img_w, img_h = img_meta.resolution
        img_area = img_w * img_h
        
        mask_areas = []
        for mask_meta in img_meta.masks:
            mask_path = project_root / mask_meta.mask_path
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                area = cv2.countNonZero(mask_img)
                mask_areas.append({
                    "mask_id": mask_meta.mask_id,
                    "area": area,
                    "relative_area": area / img_area
                })
                
        # Sort descending by area
        mask_areas.sort(key=lambda x: x["area"], reverse=True)
        
        # Calculate cumulative coverage
        total_segmented_area = sum(m["area"] for m in mask_areas)
        
        cum_area = 0
        for i, m in enumerate(mask_areas):
            cum_area += m["area"]
            m["rank"] = i + 1
            m["cumulative_coverage"] = cum_area / total_segmented_area if total_segmented_area > 0 else 0
            m["image_id"] = image_id
            m["total_masks"] = len(mask_areas)
            data.append(m)
            
    df = pd.DataFrame(data)
    
    # 1. Histograms
    plt.figure(figsize=(10, 5))
    plt.hist(df["area"], bins=50, color='blue', alpha=0.7)
    plt.title("Absolute Area Histogram")
    plt.xlabel("Area (Pixels)")
    plt.ylabel("Frequency")
    plt.savefig(interim_dir / "area_hist.png")
    
    plt.figure(figsize=(10, 5))
    plt.hist(df["relative_area"], bins=50, color='green', alpha=0.7)
    plt.title("Relative Area Histogram")
    plt.xlabel("Relative Area (Mask Area / Image Area)")
    plt.ylabel("Frequency")
    plt.savefig(interim_dir / "relative_area_hist.png")
    
    # 2. Masks-per-image statistics
    masks_per_img = df.groupby("image_id").size()
    print("--- Masks Per Image ---")
    print(masks_per_img.describe())
    
    # 3. Cumulative Area Coverage
    print("\n--- Cumulative Area Coverage (Averaged across images) ---")
    ranks_to_check = [1, 3, 5, 10, 15, 20]
    for r in ranks_to_check:
        cov = df[df["rank"] == r]["cumulative_coverage"].mean()
        print(f"Top {r} Mask Coverage: {cov:.2%}")
        
    # How many masks for 80%, 90%, 95%?
    print("\n--- Masks required to reach coverage threshold (Mean per image) ---")
    for thresh in [0.80, 0.90, 0.95]:
        masks_needed = []
        for img_id in df["image_id"].unique():
            img_df = df[df["image_id"] == img_id]
            # find first rank where cumulative coverage >= thresh
            reach = img_df[img_df["cumulative_coverage"] >= thresh]
            if len(reach) > 0:
                masks_needed.append(reach.iloc[0]["rank"])
        print(f"Masks for {thresh*100}% coverage: {np.mean(masks_needed):.1f} masks")
        
    print("\n--- Analysis completed. ---")

if __name__ == "__main__":
    analyze()
