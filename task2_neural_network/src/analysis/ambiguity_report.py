import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.data.metadata_mgr import MetadataManager
from src.features.geometric import extract_geometric_features
from src.core.labels import StructuralClass
from src.utils.logger import get_logger

logger = get_logger(__name__)

def generate_ambiguity_report():
    project_root = Path(".")
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    mgr = MetadataManager(metadata_path)
    
    ambiguous_masks = []
    
    for img_meta in mgr.data:
        image_path = project_root / img_meta.image_path
        image = cv2.imread(str(image_path))
        if image is None: continue
        
        img_area = image.shape[0] * image.shape[1]
        
        for mask_meta in img_meta.masks:
            is_unknown = (mask_meta.structural_class == StructuralClass.UNKNOWN.value)
            is_skipped = (mask_meta.skip_reason is not None)
            
            # We track UNKNOWN labels and skipped masks as ambiguous/noise
            if is_unknown or is_skipped:
                mask_path = project_root / mask_meta.mask_path
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None: continue
                
                features = extract_geometric_features(mask, image_area=img_area)
                features["mask_id"] = mask_meta.mask_id
                features["structural_class"] = mask_meta.structural_class
                features["skip_reason"] = mask_meta.skip_reason
                features["is_skipped"] = is_skipped
                
                ambiguous_masks.append(features)
                
    df = pd.DataFrame(ambiguous_masks)
    
    if df.empty:
        logger.info("No ambiguous or skipped masks found.")
        return
        
    out_dir = project_root / "data" / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Count
    total_ambiguous = len(df)
    total_skipped = df["is_skipped"].sum()
    total_unknown = total_ambiguous - total_skipped
    
    # 2. Average geometry
    avg_geom = df[["area", "relative_area", "solidity", "extent", "vertex_count"]].mean().round(4)
    
    # 3. Histogram of relative area
    plt.figure(figsize=(10, 6))
    sns.histplot(df["relative_area"], bins=50, kde=True, color="orange")
    plt.title("Distribution of Relative Area for Ambiguous/Skipped Masks")
    plt.xlabel("Relative Area")
    plt.ylabel("Frequency")
    plt.tight_layout()
    hist_path = out_dir / "ambiguity_relative_area_hist.png"
    plt.savefig(hist_path)
    plt.close()
    
    # 4. Top recurring patterns (e.g. Tiny and Solid vs Large and Complex)
    # Categorize into simple patterns
    conditions = [
        (df["relative_area"] < 0.002) & (df["solidity"] >= 0.8),
        (df["relative_area"] < 0.002) & (df["solidity"] < 0.8),
        (df["relative_area"] >= 0.002) & (df["vertex_count"] > 15),
        (df["relative_area"] >= 0.002) & (df["vertex_count"] <= 15)
    ]
    choices = [
        "Tiny & Solid (Noise/Dots)",
        "Tiny & Irregular (Noise)",
        "Large & Complex (Text/Graphics)",
        "Large & Simple (Potential Ontology Confusion)"
    ]
    df["pattern"] = np.select(conditions, choices, default="Other")
    pattern_counts = df["pattern"].value_counts()
    
    report_path = out_dir / "ambiguity_report.md"
    with open(report_path, "w") as f:
        f.write("# Ambiguity Tracking Report\n\n")
        f.write(f"**Total Ambiguous / Skipped Masks:** {total_ambiguous}\n")
        f.write(f"- Skipped by area/rank: {total_skipped}\n")
        f.write(f"- Labeled as UNKNOWN: {total_unknown}\n\n")
        
        f.write("## Average Geometric Profile\n")
        f.write("| Feature | Average |\n")
        f.write("|---|---|\n")
        for k, v in avg_geom.items():
            f.write(f"| {k} | {v} |\n")
        f.write("\n")
        
        f.write("## Top Recurring Ambiguity Patterns\n")
        f.write("| Pattern | Count | Percentage |\n")
        f.write("|---|---|---|\n")
        for pattern, count in pattern_counts.items():
            f.write(f"| {pattern} | {count} | {count/total_ambiguous*100:.1f}% |\n")
            
        f.write("\n## Relative Area Distribution\n")
        f.write("![Relative Area Histogram](ambiguity_relative_area_hist.png)\n")
        
    logger.info(f"Generated Ambiguity Report at {report_path}")

if __name__ == "__main__":
    generate_ambiguity_report()
