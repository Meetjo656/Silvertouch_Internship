import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.data.metadata_mgr import MetadataManager
from src.features.geometric import extract_geometric_features
from src.core.labels import StructuralClass

def generate_report():
    project_root = Path(".")
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    mgr = MetadataManager(metadata_path)
    
    data = []
    
    for img_meta in mgr.data:
        image_path = project_root / img_meta.image_path
        image = cv2.imread(str(image_path))
        if image is None: continue
        
        img_area = image.shape[0] * image.shape[1]
        
        for mask_meta in img_meta.masks:
            # Only analyze labeled masks, ignore UNKNOWN
            if mask_meta.structural_class in [StructuralClass.UNKNOWN.value, "unlabeled"]:
                continue
                
            mask_path = project_root / mask_meta.mask_path
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None: continue
            
            features = extract_geometric_features(mask, image_area=img_area)
            features["structural_class"] = mask_meta.structural_class
            features["image_id"] = img_meta.image_id
            features["mask_id"] = mask_meta.mask_id
            
            data.append(features)
            
    df = pd.DataFrame(data)
    
    out_dir = project_root / "data" / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if len(df) == 0:
        print("No labeled data found to generate feature report.")
        return
        
    # Save CSV
    df.to_csv(out_dir / "structural_features.csv", index=False)
    
    # Generate Markdown Report
    feature_cols = [c for c in df.columns if c not in ["structural_class", "image_id", "mask_id"]]
    stats = df.groupby("structural_class")[feature_cols].mean().round(4)
    
    with open(out_dir / "structural_features_report.md", "w") as f:
        f.write("# Structural Feature Report\n\n")
        f.write(f"Total labeled masks analyzed: {len(df)}\n\n")
        f.write("## Per-Class Averages\n\n")
        f.write(stats.to_markdown())
        
    # Generate Visual Plots
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="structural_class", y="area", data=df)
    plt.title("Distribution of Area by Structural Class")
    plt.savefig(out_dir / "feature_plot_area.png")
    plt.close()

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x="solidity", y="extent", hue="structural_class", data=df)
    plt.title("Solidity vs Extent by Structural Class")
    plt.savefig(out_dir / "feature_plot_solidity_extent.png")
    plt.close()

    print("Successfully generated structural feature reports and plots.")

if __name__ == "__main__":
    generate_report()
