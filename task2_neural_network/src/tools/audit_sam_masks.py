import cv2
import numpy as np
from pathlib import Path
from src.data.metadata_mgr import MetadataManager

def overlay_all_masks(image: np.ndarray, masks: list, alpha: float = 0.5) -> np.ndarray:
    overlay = image.copy()
    np.random.seed(42) # For consistent colors
    for mask in masks:
        color = np.random.randint(0, 255, (3,)).tolist()
        bool_mask = mask > 127
        overlay[bool_mask] = color
    
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image

def run_audit():
    project_root = Path(".")
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    mgr = MetadataManager(metadata_path)
    
    out_dir = project_root / "data" / "interim" / "audit_reviews"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    for img_meta in mgr.data:
        image_id = img_meta.image_id
        image_path = project_root / img_meta.image_path
        image = cv2.imread(str(image_path))
        if image is None: continue
        
        masks = []
        for mask_meta in img_meta.masks:
            mask_path = project_root / mask_meta.mask_path
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                area = cv2.countNonZero(mask)
                if area > 10:
                    masks.append(mask)
                    
        # Create side-by-side: Original | All Masks Overlaid
        overlaid = overlay_all_masks(image.copy(), masks)
        
        # Resize to fixed height for easier viewing side-by-side
        target_h = 600
        aspect = image.shape[1] / image.shape[0]
        target_w = int(target_h * aspect)
        
        img_resized = cv2.resize(image, (target_w, target_h))
        over_resized = cv2.resize(overlaid, (target_w, target_h))
        
        combined = np.hstack((img_resized, over_resized))
        
        # Add text labels
        cv2.putText(combined, "Original Image (Visible Creases)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(combined, "SAM Segmentation Output", (target_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out_path = out_dir / f"audit_{image_id}.jpg"
        cv2.imwrite(str(out_path), combined)
        generated_files.append(str(out_path))
        print(f"Generated audit for {image_id} with {len(masks)} masks.")

if __name__ == "__main__":
    run_audit()
