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

def draw_text(img, text, pos, font_scale=0.6, thickness=2):
    x, y = pos
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(img, (x, y - th - 5), (x + tw, y + 5), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

def generate_review():
    project_root = Path(".")
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    mgr = MetadataManager(metadata_path)
    
    out_dir = project_root / "data" / "interim" / "candidate_reviews"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    top_n = 20
    cols = 5
    rows = math.ceil(top_n / cols)
    cell_w, cell_h = 400, 400
    
    generated_files = []

    for img_meta in mgr.data:
        image_id = img_meta.image_id
        image_path = project_root / img_meta.image_path
        image = cv2.imread(str(image_path))
        if image is None: 
            print(f"Failed to load image: {image_path}")
            continue
            
        img_area = image.shape[0] * image.shape[1]
        
        # Load all masks and compute area
        masks_info = []
        for mask_meta in img_meta.masks:
            mask_path = project_root / mask_meta.mask_path
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                area = cv2.countNonZero(mask)
                masks_info.append({"mask": mask, "area": area, "mask_id": mask_meta.mask_id})
                
        # Sort by descending area
        masks_info.sort(key=lambda x: x["area"], reverse=True)
        top_masks = masks_info[:top_n]
        
        # Create grid
        grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        
        for idx, m_info in enumerate(top_masks):
            r = idx // cols
            c = idx % cols
            
            mask = m_info["mask"]
            area = m_info["area"]
            rel_area = area / img_area
            
            overlaid = overlay_mask(image.copy(), mask)
            resized = cv2.resize(overlaid, (cell_w, cell_h))
            
            # Annotate
            draw_text(resized, f"Rank: {idx + 1}", (10, 30))
            draw_text(resized, f"Area: {area}", (10, 60))
            draw_text(resized, f"Rel: {rel_area*100:.2f}%", (10, 90))
            
            grid[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = resized
            
        out_path = out_dir / f"review_{image_id}.jpg"
        cv2.imwrite(str(out_path), grid)
        generated_files.append(str(out_path))
        print(f"Generated review for {image_id}")
        
    print(f"\nSuccessfully generated {len(generated_files)} review images.")

if __name__ == "__main__":
    generate_review()
