"""
Manual Labeling GUI.

A lightweight OpenCV-based tool for annotating SAM masks as structural or 
non-structural components.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import List

from src.data.metadata_mgr import MetadataManager
from src.core.labels import StructuralClass
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MaskLabeler:
    """Provides a simple OpenCV GUI for labeling masks."""
    
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        self.metadata_path = self.project_root / "data" / "metadata" / "dataset_metadata.json"
        self.metadata_mgr = MetadataManager(self.metadata_path)
        self.annotator_name = "default_user"
        
    def overlay_mask(self, image: np.ndarray, mask: np.ndarray, color: tuple = (0, 255, 0), alpha: float = 0.5) -> np.ndarray:
        """Overlays a binary mask onto an image with a specific color."""
        overlay = image.copy()
        
        # Ensure mask is boolean
        bool_mask = mask > 127
        
        # Apply color to the masked area on the overlay
        overlay[bool_mask] = color
        
        # Blend the overlay with the original image
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        return image

    def run(self, annotator: str):
        """Starts the labeling session."""
        self.annotator_name = annotator
        
        if not self.metadata_mgr.data:
            logger.error("No metadata found. Please ingest images and masks first.")
            return
            
        print("\n--- Mask Labeling Instructions ---")
        print("Press '1' for PANEL")
        print("Press '2' for FLAP")
        print("Press '3' for GLUE_TAB")
        print("Press '4' for DUST_FLAP")
        print("Press '5' for BACKGROUND")
        print("Press '0' for UNKNOWN")
        print("Press 'Space' to skip")
        print("Press 'q' to quit and save")
        print("----------------------------------\n")
        
        total_images = len(self.metadata_mgr.data)
        total_candidates = sum(1 for img in self.metadata_mgr.data for m in img.masks if m.is_candidate)
        
        for img_idx, img_meta in enumerate(self.metadata_mgr.data):
            img_path = self.project_root / img_meta.image_path
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue
                
            image = cv2.imread(str(img_path))
            if image is None:
                continue
                
            for mask_idx, mask_meta in enumerate(img_meta.masks):
                # Skip if it is not a candidate
                if not getattr(mask_meta, "is_candidate", True):
                    continue
                    
                if mask_meta.structural_class != StructuralClass.UNKNOWN.value:
                    continue
                    
                mask_path = self.project_root / mask_meta.mask_path
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                if mask is None:
                    continue
                    
                display_img = image.copy()
                display_img = self.overlay_mask(display_img, mask, color=(0, 255, 0)) # Green overlay
                
                # Compute remaining candidates dynamically
                labeled_candidates = sum(1 for img in self.metadata_mgr.data for m in img.masks if m.is_candidate and m.structural_class != StructuralClass.UNKNOWN.value)
                candidates_remaining = total_candidates - labeled_candidates
                
                # Add text info
                info_text = f"Image {img_idx+1}/{total_images} | Mask {mask_idx+1}/{len(img_meta.masks)} | Candidates Remaining: {candidates_remaining}"
                cv2.putText(display_img, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(display_img, "1:PANEL 2:FLAP 3:GLUE 4:DUST 5:BG 0:UNK | Space:Skip q:Quit", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                
                # Resize for display if too large
                h, w = display_img.shape[:2]
                max_dim = 1000
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    display_img = cv2.resize(display_img, (int(w * scale), int(h * scale)))
                
                cv2.imshow("Mask Labeler", display_img)
                
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    assigned_class = None
                    if key == ord('1'): assigned_class = StructuralClass.PANEL.value
                    elif key == ord('2'): assigned_class = StructuralClass.FLAP.value
                    elif key == ord('3'): assigned_class = StructuralClass.GLUE_TAB.value
                    elif key == ord('4'): assigned_class = StructuralClass.DUST_FLAP.value
                    elif key == ord('5'): assigned_class = StructuralClass.BACKGROUND.value
                    elif key == ord('0'): assigned_class = StructuralClass.UNKNOWN.value
                    
                    if assigned_class is not None:
                        self.metadata_mgr.update_mask_label(img_meta.image_id, mask_meta.mask_id, assigned_class, self.annotator_name)
                        # Visual confirmation
                        feedback_img = display_img.copy()
                        cv2.putText(feedback_img, f"Assigned: {assigned_class}", (int(w/2) - 100, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                        cv2.imshow("Mask Labeler", feedback_img)
                        cv2.waitKey(300) # brief pause to show feedback
                        break
                    elif key == ord(' '):
                        print(f"Skipping {mask_meta.mask_id}")
                        break
                    elif key == ord('q'):
                        print("Quitting...")
                        cv2.destroyAllWindows()
                        return
                        
        cv2.destroyAllWindows()
        print("Labeling complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask Labeling Tool")
    parser.add_argument("--project_root", type=str, default=".", help="Root directory of the project")
    parser.add_argument("--annotator", type=str, default="expert", help="Name of annotator")
    args = parser.parse_args()
    
    labeler = MaskLabeler(args.project_root)
    labeler.run(args.annotator)
