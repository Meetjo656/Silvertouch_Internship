"""
Dataset Builder.

This module provides functionalities to build and load the dataset from the 
metadata and feature stores for training the machine learning models.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import cv2

from src.data.metadata_mgr import MetadataManager
from src.core.labels import StructuralClass
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MaskDatasetBuilder:
    """Loads and formats mask data for training the baseline model."""
    
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        self.metadata_path = self.project_root / "data" / "metadata" / "dataset_metadata.json"
        self.metadata_mgr = MetadataManager(self.metadata_path)
        
    def load_labeled_masks(self) -> Tuple[List[str], List[np.ndarray], List[str], List[str], List[float]]:
        """
        Scans metadata for all labeled masks and loads their corresponding 
        binary masks from disk.
        
        Returns:
            Tuple containing:
                - List of mask IDs
                - List of numpy arrays representing binary masks (2D)
                - List of string labels (e.g., "PANEL", "FLAP")
                - List of image IDs (for grouping)
                - List of image areas (for relative area calculation)
        """
        mask_ids = []
        masks = []
        labels = []
        image_ids = []
        image_areas = []
        
        if not self.metadata_mgr.data:
            logger.warning("Metadata is empty. No labeled masks to load.")
            return mask_ids, masks, labels, image_ids, image_areas
            
        for img_meta in self.metadata_mgr.data:
            img_area = float(img_meta.resolution[0] * img_meta.resolution[1])
            for mask_meta in img_meta.masks:
                if mask_meta.structural_class not in [StructuralClass.UNKNOWN.value, "unlabeled"]:
                    # Load mask image
                    mask_full_path = self.project_root / mask_meta.mask_path
                    if not mask_full_path.exists():
                        logger.warning(f"Mask file not found: {mask_full_path}. Skipping.")
                        continue
                        
                    # Read as grayscale binary
                    mask_img = cv2.imread(str(mask_full_path), cv2.IMREAD_GRAYSCALE)
                    if mask_img is None:
                        logger.warning(f"Failed to read mask image: {mask_full_path}. Skipping.")
                        continue
                        
                    # Ensure binary (0 and 1) robustly
                    # This handles both 0/1 binary masks and 0/255 uint8 masks
                    binary_mask = (mask_img > 0).astype(np.uint8)
                    mask_ids.append(mask_meta.mask_id)
                    masks.append(binary_mask)
                    labels.append(mask_meta.structural_class)
                    image_ids.append(img_meta.image_id)
                    image_areas.append(img_area)
                    
        logger.info(f"Loaded {len(mask_ids)} labeled masks for training/evaluation.")
        return mask_ids, masks, labels, image_ids, image_areas
