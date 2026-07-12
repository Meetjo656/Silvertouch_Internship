import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.data.metadata_mgr import MetadataManager
from src.core.labels import StructuralClass
from src.utils.logger import get_logger

logger = get_logger(__name__)

def generate_dashboard(project_root: str | Path):
    """Generates the dataset progress dashboard and distribution chart."""
    project_root = Path(project_root)
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    
    if not metadata_path.exists():
        logger.warning(f"Metadata file not found at {metadata_path}. Skipping dashboard generation.")
        return
        
    mgr = MetadataManager(metadata_path)
    
    total_masks = 0
    labeled_masks = 0
    candidates_remaining = 0
    removed_by_floor = 0
    total_candidates = 0
    
    class_counts = {
        "PANEL": 0,
        "FLAP": 0,
        "GLUE_TAB": 0,
        "DUST_FLAP": 0,
        "BACKGROUND": 0,
        "UNKNOWN": 0
    }
    
    for img_meta in mgr.data:
        for mask_meta in img_meta.masks:
            total_masks += 1
            cls_name = mask_meta.structural_class
            
            if getattr(mask_meta, "skip_reason", None) == "below_min_relative_area":
                removed_by_floor += 1
                
            if getattr(mask_meta, "is_candidate", True):
                total_candidates += 1
                if cls_name == "UNKNOWN":
                    candidates_remaining += 1
                    
                if cls_name in class_counts:
                    class_counts[cls_name] += 1
                else:
                    class_counts["UNKNOWN"] += 1
                    
                if cls_name != "UNKNOWN":
                    labeled_masks += 1
                
    unlabeled_masks = total_masks - labeled_masks
    completion_percentage = (labeled_masks / total_candidates * 100) if total_candidates > 0 else 0.0
    unknown_percentage = (class_counts["UNKNOWN"] / total_candidates * 100) if total_candidates > 0 else 0.0
    
    # 1. Output directory
    out_dir = project_root / "data" / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Write Markdown Report
    report_path = out_dir / "dataset_progress_report.md"
    with open(report_path, "w") as f:
        f.write("# Dataset Progress Dashboard\n\n")
        f.write("## Candidate Funnel\n")
        f.write(f"- **Total Raw Masks:** {total_masks}\n")
        f.write(f"- **Removed by Area Floor:** {removed_by_floor}\n")
        f.write(f"- **Total Candidates:** {total_candidates}\n")
        f.write(f"- **Candidates Remaining (Unlabeled):** {candidates_remaining}\n\n")
        
        f.write("## Overall Progress\n")
        f.write(f"- **Labeled Candidate Masks:** {labeled_masks}\n")
        f.write(f"- **Annotation Completion:** {completion_percentage:.2f}%\n")
        f.write(f"- **UNKNOWN Percentage (of candidates):** {unknown_percentage:.2f}%\n\n")
        
        f.write("## Class Distribution\n")
        f.write("| Class | Count |\n")
        f.write("|---|---|\n")
        for cls_name in ["PANEL", "FLAP", "GLUE_TAB", "DUST_FLAP", "BACKGROUND", "UNKNOWN"]:
            f.write(f"| {cls_name} | {class_counts[cls_name]} |\n")
            
        f.write("\n![Class Distribution](class_distribution.png)\n")
        
    logger.info(f"Generated Dataset Progress Dashboard at {report_path}")
    
    # 3. Generate Chart
    plt.figure(figsize=(10, 6))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    sns.barplot(x=classes, y=counts, hue=classes, palette="viridis", legend=False)
    plt.title(f"Class Distribution (Total Labeled: {labeled_masks}/{total_masks})")
    plt.ylabel("Count")
    plt.xlabel("Structural Class")
    plt.xticks(rotation=45)
    
    # Add count labels on top of bars
    for i, count in enumerate(counts):
        plt.text(i, count + (max(counts) * 0.01), str(count), ha='center', va='bottom')
        
    plt.tight_layout()
    chart_path = out_dir / "class_distribution.png"
    plt.savefig(chart_path)
    plt.close()
    logger.info(f"Generated Class Distribution Chart at {chart_path}")

if __name__ == "__main__":
    generate_dashboard(".")
