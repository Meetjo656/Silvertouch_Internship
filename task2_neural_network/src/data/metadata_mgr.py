"""
Metadata management system for the Structural Mask Classifier.

This module defines the JSON schema for dataset metadata using Pydantic
and provides a manager class to handle loading, saving, and updating
annotations for masks.
"""

import json
from pathlib import Path
from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, validator
from datetime import datetime

from src.utils.logger import get_logger
from src.core.labels import StructuralClass

logger = get_logger(__name__)

class MaskMetadata(BaseModel):
    """Schema for individual mask metadata."""
    mask_id: str
    mask_path: str
    structural_class: str = StructuralClass.UNKNOWN.value
    annotator: Optional[str] = None
    timestamp: Optional[str] = None
    is_candidate: bool = True
    skip_reason: Optional[str] = None

    @validator("structural_class")
    def validate_structural_class(cls, v):
        valid_classes = [c.value for c in StructuralClass]
        if v not in valid_classes:
            raise ValueError(f"Invalid structural_class: {v}. Must be one of {valid_classes}")
        return v

class ImageMetadata(BaseModel):
    """Schema for image-level metadata containing multiple masks."""
    image_id: str
    image_path: str
    resolution: Tuple[int, int]
    masks: List[MaskMetadata] = Field(default_factory=list)

class MetadataManager:
    """Handles CRUD operations for the dataset metadata JSON file."""
    
    def __init__(self, metadata_path: str | Path):
        """
        Initializes the MetadataManager.
        
        Args:
            metadata_path (str | Path): Path to the JSON metadata file.
        """
        self.metadata_path = Path(metadata_path)
        self.data: List[ImageMetadata] = []
        self._load()

    def _load(self) -> None:
        """Loads metadata from the JSON file if it exists."""
        if not self.metadata_path.exists():
            logger.info(f"Metadata file not found at {self.metadata_path}. Initializing empty.")
            return

        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                
            # Auto-upgrade logic for migration from binary 'label' to 'structural_class'
            for img_data in raw_data:
                for mask_data in img_data.get('masks', []):
                    if 'label' in mask_data:
                        mask_data.pop('label')
                    if 'structural_class' not in mask_data:
                        mask_data['structural_class'] = StructuralClass.UNKNOWN.value

            self.data = [ImageMetadata(**item) for item in raw_data]
            logger.info(f"Successfully loaded metadata with {len(self.data)} images.")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            raise

    def save(self) -> None:
        """Saves current metadata to the JSON file."""
        try:
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json_data = [item.model_dump() for item in self.data]
                json.dump(json_data, f, indent=4)
            logger.info(f"Successfully saved metadata to {self.metadata_path}.")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise

    def add_image(self, image_metadata: ImageMetadata) -> None:
        """
        Adds a new image record or overwrites an existing one.
        
        Args:
            image_metadata (ImageMetadata): The image metadata object.
        """
        # Remove existing if present
        self.data = [img for img in self.data if img.image_id != image_metadata.image_id]
        self.data.append(image_metadata)
        logger.debug(f"Added/Updated metadata for image {image_metadata.image_id}.")
        self.save()

    def update_mask_label(
        self, image_id: str, mask_id: str, structural_class: str, annotator: str
    ) -> bool:
        """
        Updates the structural_class for a specific mask.
        
        Args:
            image_id (str): The ID of the image.
            mask_id (str): The ID of the mask.
            structural_class (str): The new structural_class.
            annotator (str): Name of the annotator.
            
        Returns:
            bool: True if mask was found and updated, False otherwise.
        """
        for img in self.data:
            if img.image_id == image_id:
                for mask in img.masks:
                    if mask.mask_id == mask_id:
                        mask.structural_class = structural_class
                        mask.annotator = annotator
                        mask.timestamp = datetime.utcnow().isoformat() + "Z"
                        logger.info(f"Updated structural_class for {mask_id} to '{structural_class}'.")
                        self.save()
                        return True
        logger.warning(f"Mask {mask_id} not found in image {image_id}.")
        return False
