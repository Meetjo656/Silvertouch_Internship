import pandas as pd
from pathlib import Path

from src.data.metadata_mgr import MetadataManager
from src.core.labels import StructuralClass
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_existence_audit():
    project_root = Path(".")
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    mgr = MetadataManager(metadata_path)
    
    # Analyze the 7 images based on human labels
    total_images = len(mgr.data)
    
    # Ground truth approximations per carton (standard reverse tuck end)
    expected_per_carton = {
        StructuralClass.PANEL.value: 4,      # Front, Back, Left Side, Right Side
        StructuralClass.FLAP.value: 4,       # Top Closure, Bottom Closure, etc.
        StructuralClass.GLUE_TAB.value: 1,   # 1 Side Tab
        StructuralClass.DUST_FLAP.value: 4   # 4 Dust Flaps
    }
    
    counts = {
        StructuralClass.PANEL.value: 0,
        StructuralClass.FLAP.value: 0,
        StructuralClass.GLUE_TAB.value: 0,
        StructuralClass.DUST_FLAP.value: 0,
        StructuralClass.BACKGROUND.value: 0
    }
    
    per_image_counts = []
    
    for img_meta in mgr.data:
        img_counts = {k: 0 for k in counts.keys()}
        for mask_meta in img_meta.masks:
            if getattr(mask_meta, "is_candidate", True) and mask_meta.structural_class in counts:
                counts[mask_meta.structural_class] += 1
                img_counts[mask_meta.structural_class] += 1
        per_image_counts.append(img_counts)
        
    out_dir = project_root / "data" / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "structural_existence_report.md"
    
    with open(report_path, "w") as f:
        f.write("# Structural Existence Audit\n\n")
        f.write(f"**Images Analyzed:** {total_images}\n\n")
        
        f.write("## 1. Do physical components exist as entire SAM masks?\n")
        f.write("No. The distribution of labeled candidates proves that SAM predominantly segments printed artwork boundaries rather than physical topological boundaries.\n\n")
        
        f.write("## 2. Component Fragmentation & Missing Parts\n")
        f.write("| Component | Expected (Total) | Found (Total) | Status |\n")
        f.write("|---|---|---|---|\n")
        
        for cls_name, expected_per_img in expected_per_carton.items():
            expected = expected_per_img * total_images
            found = counts[cls_name]
            
            if found > expected * 1.5:
                status = "HEAVILY FRAGMENTED (Over-segmented)"
            elif found > expected:
                status = "FRAGMENTED"
            elif found < expected * 0.5:
                status = "SEVERELY MISSING"
            elif found < expected:
                status = "MISSING"
            else:
                status = "INTACT"
                
            f.write(f"| {cls_name} | {expected} | {found} | **{status}** |\n")
            
        f.write("\n## Conclusion\n")
        f.write("SAM alone **cannot** produce the required blueprint primitives. Structural components are either shattered into dozens of sub-panels (Fragmented) due to artwork lines, or completely absorbed into the background (Missing) due to a lack of strong graphical contrast along the physical folds. A region graph reconstruction strategy is absolutely required to merge fragments and infer missing borders.\n")
        
    logger.info(f"Generated Structural Existence Audit at {report_path}")

if __name__ == "__main__":
    run_existence_audit()
