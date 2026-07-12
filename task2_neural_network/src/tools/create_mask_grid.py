import cv2
import numpy as np
import math
from pathlib import Path
from src.data.metadata_mgr import MetadataManager

def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple = (0, 255, 0), alpha: float = 0.5) -> np.ndarray:
    overlay = image.copy()
    bool_mask = mask > 127
    overlay[bool_mask] = color
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image

def create_grid(num_masks=36, cols=6):
    project_root = Path(".")
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    mgr = MetadataManager(metadata_path)
    
    candidates = []
    for img_meta in mgr.data:
        image_path = project_root / img_meta.image_path
        image = cv2.imread(str(image_path))
        if image is None: continue
        
        for mask_meta in img_meta.masks:
            if getattr(mask_meta, "is_candidate", True) and mask_meta.label == "unlabeled":
                mask_path = project_root / mask_meta.mask_path
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    candidates.append((image, mask, mask_meta.mask_id))
                    if len(candidates) >= num_masks:
                        break
        if len(candidates) >= num_masks:
            break

    if not candidates:
        print("No candidates found.")
        return

    # Resize all to a fixed size for the grid
    cell_h, cell_w = 400, 400
    rows = math.ceil(len(candidates) / cols)
    
    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    
    for idx, (img, mask, mask_id) in enumerate(candidates):
        r = idx // cols
        c = idx % cols
        
        # Crop to bounding box to see it better? Or resize whole image?
        # Let's crop around the mask
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # Add padding
            pad = 50
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(img.shape[1] - x, w + 2*pad)
            h = min(img.shape[0] - y, h + 2*pad)
            
            cropped_img = img[y:y+h, x:x+w]
            cropped_mask = mask[y:y+h, x:x+w]
        else:
            cropped_img = img
            cropped_mask = mask
            
        overlaid = overlay_mask(cropped_img.copy(), cropped_mask)
        
        # Resize to cell
        resized = cv2.resize(overlaid, (cell_w, cell_h))
        
        # Add text (Index 1 to N)
        cv2.putText(resized, f"ID: {idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        grid[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = resized

    out_path = project_root / "data" / "interim" / "mask_grid.jpg"
    cv2.imwrite(str(out_path), grid)
    print(f"Saved grid with {len(candidates)} masks to {out_path}")
    
    # Save a map of ID -> mask_id to use later
    with open("data/interim/mask_grid_map.txt", "w") as f:
        for idx, (_, _, mask_id) in enumerate(candidates):
            f.write(f"{idx}:{mask_id}\n")

if __name__ == "__main__":
    create_grid(42, 6) # 42 masks (7x6)

