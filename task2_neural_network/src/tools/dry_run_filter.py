import cv2
import numpy as np
import math
from pathlib import Path
from src.data.metadata_mgr import MetadataManager
from src.features.geometric import extract_geometric_features

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

def dry_run():
    project_root = Path(".")
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    mgr = MetadataManager(metadata_path)
    
    out_dir = project_root / "data" / "interim" / "dry_run_reviews"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    total_original = 0
    total_retained = 0
    
    cols = 4
    cell_w, cell_h = 400, 400
    
    generated_files = []
    
    for img_meta in mgr.data:
        image_id = img_meta.image_id
        image_path = project_root / img_meta.image_path
        image = cv2.imread(str(image_path))
        if image is None: continue
        
        img_area = image.shape[0] * image.shape[1]
        
        # Load masks and calculate area
        masks_list = []
        for mask_meta in img_meta.masks:
            mask_path = project_root / mask_meta.mask_path
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                area = cv2.countNonZero(mask)
                masks_list.append({
                    "mask_id": mask_meta.mask_id,
                    "mask": mask,
                    "area": area
                })
        
        total_orig_img = len(masks_list)
        total_original += total_orig_img
        
        # Sort by area
        masks_list.sort(key=lambda x: x["area"], reverse=True)
        
        retained = []
        for rank, m in enumerate(masks_list):
            if rank >= 8: # Top 8 ceiling
                continue
                
            features = extract_geometric_features(m["mask"], image_area=img_area)
            rel_area = features["relative_area"]
            solidity = features["solidity"]
            extent = features["extent"]
            
            if rel_area >= 0.015 and solidity >= 0.50 and extent >= 0.40:
                retained.append({
                    "mask": m["mask"],
                    "rank": rank + 1,
                    "rel_area": rel_area,
                    "solidity": solidity,
                    "extent": extent
                })
                
        total_ret_img = len(retained)
        total_retained += total_ret_img
        
        print(f"Image {image_id}: Original = {total_orig_img}, Retained = {total_ret_img}")
        
        if total_ret_img == 0:
            continue
            
        rows = math.ceil(total_ret_img / cols)
        grid = np.zeros((max(1, rows) * cell_h, cols * cell_w, 3), dtype=np.uint8)
        
        for idx, r_info in enumerate(retained):
            r = idx // cols
            c = idx % cols
            
            overlaid = overlay_mask(image.copy(), r_info["mask"])
            resized = cv2.resize(overlaid, (cell_w, cell_h))
            
            draw_text(resized, f"Rank: {r_info['rank']}", (10, 30))
            draw_text(resized, f"Rel: {r_info['rel_area']*100:.2f}%", (10, 60))
            draw_text(resized, f"Sol: {r_info['solidity']:.2f}", (10, 90))
            draw_text(resized, f"Ext: {r_info['extent']:.2f}", (10, 120))
            
            grid[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = resized
            
        out_path = out_dir / f"dry_run_{image_id}.jpg"
        cv2.imwrite(str(out_path), grid)
        generated_files.append(str(out_path))
        
    print("\n--- Dry Run Summary ---")
    print(f"Total Original Masks: {total_original}")
    print(f"Total Retained Candidates: {total_retained}")
    retention_pct = (total_retained / total_original * 100) if total_original > 0 else 0
    print(f"Retention Percentage: {retention_pct:.2f}%")

if __name__ == "__main__":
    dry_run()
