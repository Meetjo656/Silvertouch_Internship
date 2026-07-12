"""
Dataset Validation Script.

Performs integrity checks on the dataset metadata and physical files
before baseline training, as outlined in Phase B of the experiment plan.
"""

import argparse
from pathlib import Path
from collections import Counter
import cv2

from src.data.metadata_mgr import MetadataManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

def validate_dataset(project_root: str):
    """Runs all dataset validation checks."""
    project_root = Path(project_root)
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    
    logger.info("Starting Dataset Validation...")
    
    if not metadata_path.exists():
        logger.error(f"Metadata file not found at {metadata_path}")
        return False
        
    metadata_mgr = MetadataManager(metadata_path)
    if not metadata_mgr.data:
        logger.error("Metadata is empty. Please run ingestion and labeling workflows.")
        return False

    all_passed = True
    
    # 1. Class Balance Check
    logger.info("--- Class Balance Check ---")
    labels = []
    labeled_count = 0
    total_masks = 0
    
    for img_meta in metadata_mgr.data:
        for mask in img_meta.masks:
            total_masks += 1
            if mask.label != "unlabeled":
                labels.append(mask.label)
                labeled_count += 1
                
    label_counts = Counter(labels)
    logger.info(f"Total masks: {total_masks} | Labeled masks: {labeled_count}")
    for label, count in label_counts.items():
        logger.info(f"  - {label}: {count}")
        
    if label_counts.get("structural", 0) < 30:
        logger.warning("FAILED: Fewer than 30 structural masks labeled. Please label more flaps/panels to establish a baseline.")
        all_passed = False
    else:
        logger.info("PASSED: Sufficient class balance.")
        
    # 2. Missing Metadata & Duplicate Check
    logger.info("--- Metadata Integrity Check ---")
    seen_mask_ids = set()
    missing_metadata_count = 0
    duplicate_count = 0
    
    for img_meta in metadata_mgr.data:
        for mask in img_meta.masks:
            if mask.label != "unlabeled":
                if mask.mask_id in seen_mask_ids:
                    duplicate_count += 1
                    logger.warning(f"Duplicate mask ID found: {mask.mask_id}")
                seen_mask_ids.add(mask.mask_id)
                
                if not mask.annotator or not mask.timestamp:
                    missing_metadata_count += 1
                    
    if duplicate_count > 0 or missing_metadata_count > 0:
        logger.warning(f"FAILED: Found {duplicate_count} duplicates and {missing_metadata_count} masks missing annotator/timestamp.")
        all_passed = False
    else:
        logger.info("PASSED: No missing metadata or duplicates found.")

    # 3. Corrupted Mask Check
    logger.info("--- Corrupted Mask Check ---")
    corrupted_count = 0
    
    for img_meta in metadata_mgr.data:
        for mask in img_meta.masks:
            if mask.label != "unlabeled":
                mask_path = project_root / mask.mask_path
                if not mask_path.exists():
                    logger.warning(f"Mask file missing: {mask_path}")
                    corrupted_count += 1
                    continue
                    
                img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    logger.warning(f"Unreadable mask file: {mask_path}")
                    corrupted_count += 1
                    continue
                    
                if img.max() == 0:
                    logger.warning(f"Empty (all black) mask file: {mask_path}")
                    corrupted_count += 1
                    
    if corrupted_count > 0:
        logger.warning(f"FAILED: Found {corrupted_count} corrupted or missing mask files.")
        all_passed = False
    else:
        logger.info("PASSED: All labeled mask files are valid.")
        
    logger.info("---------------------------")
    if all_passed:
        logger.info("SUCCESS: Dataset validation passed. You are ready to train!")
    else:
        logger.error("FAILURE: Dataset validation failed. Please fix the warnings above before training.")
        
    return all_passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Dataset")
    parser.add_argument("--project_root", type=str, default=".", help="Project Root Directory")
    args = parser.parse_args()
    
    validate_dataset(args.project_root)
