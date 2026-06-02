import pandas as pd
from pathlib import Path
import json

from src.data.metadata_mgr import MetadataManager
from src.core.labels import StructuralClass
from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_taxonomy():
    project_root = Path(".")
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    mgr = MetadataManager(metadata_path)
    
    total_presented = 0
    unknown_count = 0
    labeled_counts = {
        StructuralClass.PANEL.value: 0,
        StructuralClass.FLAP.value: 0,
        StructuralClass.GLUE_TAB.value: 0,
        StructuralClass.DUST_FLAP.value: 0,
        StructuralClass.BACKGROUND.value: 0
    }
    
    for img_meta in mgr.data:
        for mask_meta in img_meta.masks:
            # We only evaluate masks that passed the candidate filter and skip reasons
            if getattr(mask_meta, "is_candidate", True) and not getattr(mask_meta, "skip_reason", None):
                # But wait, did the user label it?
                # If it's UNKNOWN but hasn't been presented to the annotator yet, we don't know if it's ambiguous or just unlabeled.
                # Since we don't have a separate "labeled=True" flag, we assume that any mask with an annotator name is "labeled".
                # If annotator is None, it hasn't been seen by a human yet, so skip.
                if getattr(mask_meta, "annotator", None) is not None:
                    total_presented += 1
                    label = mask_meta.structural_class
                    if label == StructuralClass.UNKNOWN.value:
                        unknown_count += 1
                    elif label in labeled_counts:
                        labeled_counts[label] += 1
                        
    if total_presented == 0:
        logger.warning("No human annotations found. Please run the label GUI first.")
        return
        
    unknown_rate = unknown_count / total_presented
    
    # 1. Generate Taxonomy Validation Report
    out_dir = project_root / "data" / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = out_dir / "taxonomy_validation_report.md"
    with open(report_path, "w") as f:
        f.write("# Taxonomy Validation Report\n\n")
        f.write(f"- **Total Annotations Collected:** {total_presented}\n")
        f.write(f"- **UNKNOWN (Ambiguous) Annotations:** {unknown_count}\n")
        f.write(f"- **UNKNOWN Rate:** {unknown_rate * 100:.2f}%\n\n")
        
        f.write("## Component Breakdown\n")
        for cls_name, count in labeled_counts.items():
            f.write(f"- **{cls_name}:** {count}\n")
            
    # 2. Recommendation Engine Checkpoint
    recommendation = ""
    action = ""
    
    if unknown_rate < 0.10:
        recommendation = "Taxonomy Validated."
        action = "Keep current taxonomy. The structural classes are consistently recognizable by a human annotator."
    elif 0.10 <= unknown_rate <= 0.25:
        recommendation = "Collect More Labels."
        action = "The ambiguity rate is borderline. Continue annotation to see if the trend holds."
    else:
        recommendation = "Investigate Ontology Confusion."
        action = "High ambiguity detected. Recommend taxonomy simplification (e.g., merging FLAP and GLUE_TAB into FLAP_TAB) as human annotators cannot consistently distinguish the current boundaries."
        
    rec_path = out_dir / "taxonomy_recommendation.md"
    with open(rec_path, "w") as f:
        f.write("# Taxonomy Decision Recommendation\n\n")
        f.write(f"**Current UNKNOWN Rate:** {unknown_rate * 100:.2f}%\n\n")
        f.write(f"### Assessment: {recommendation}\n")
        f.write(f"**Action Plan:** {action}\n")
        
    logger.info(f"Generated Taxonomy Validation and Recommendation at {out_dir}")

if __name__ == "__main__":
    evaluate_taxonomy()
