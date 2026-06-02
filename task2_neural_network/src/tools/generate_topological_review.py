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

def generate_topo_review():
    project_root = Path(".")
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    mgr = MetadataManager(metadata_path)
    
    out_dir = project_root / "data" / "interim" / "topo_reviews"
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
        
        masks_list = []
        for mask_meta in img_meta.masks:
            mask_path = project_root / mask_meta.mask_path
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                area = cv2.countNonZero(mask)
                if area > 10:
                    coords = cv2.findNonZero(mask)
                    x, y, w, h = cv2.boundingRect(coords)
                    masks_list.append({
                        "mask_id": mask_meta.mask_id,
                        "mask": mask,
                        "area": area,
                        "x": x, "y": y, "w": w, "h": h
                    })
        
        total_orig_img = len(masks_list)
        total_original += total_orig_img
        
        retained = []
        for i, m1 in enumerate(masks_list):
            is_contained = False
            for j, m2 in enumerate(masks_list):
                if i != j:
                    if m1['x'] >= m2['x'] and m1['y'] >= m2['y'] and (m1['x']+m1['w']) <= (m2['x']+m2['w']) and (m1['y']+m1['h']) <= (m2['y']+m2['h']):
                        if m2['area'] > m1['area']:
                            is_contained = True
                            break
            if not is_contained:
                retained.append(m1)
                
        # Sort retained by area
        retained.sort(key=lambda x: x["area"], reverse=True)
        
        total_ret_img = len(retained)
        total_retained += total_ret_img
        
        print(f"Image {image_id}: Original = {total_orig_img}, Root Nodes = {total_ret_img}")
        
        if total_ret_img == 0:
            continue
            
        rows = math.ceil(total_ret_img / cols)
        grid = np.zeros((max(1, rows) * cell_h, cols * cell_w, 3), dtype=np.uint8)
        
        for idx, r_info in enumerate(retained):
            r = idx // cols
            c = idx % cols
            
            rel_area = r_info['area'] / img_area
            
            overlaid = overlay_mask(image.copy(), r_info["mask"])
            resized = cv2.resize(overlaid, (cell_w, cell_h))
            
            draw_text(resized, f"Root Rank: {idx+1}", (10, 30))
            draw_text(resized, f"Area: {r_info['area']}", (10, 60))
            draw_text(resized, f"Rel Area: {rel_area*100:.2f}%", (10, 90))
            
            grid[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = resized
            
        out_path = out_dir / f"topo_{image_id}.jpg"
        cv2.imwrite(str(out_path), grid)
        generated_files.append(str(out_path))
        
    print("\n--- Topological Dry Run Summary ---")
    print(f"Total Original Masks: {total_original}")
    print(f"Total Root Nodes: {total_retained}")

if __name__ == "__main__":
    generate_topo_review()
