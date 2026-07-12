import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from src.data.metadata_mgr import MetadataManager
from src.features.geometric import extract_geometric_features

def compute_adjacency(masks_data, dilate_pixels=15):
    """
    Computes how many other masks each mask is adjacent to.
    masks_data: list of dicts with 'mask' (np.ndarray) and 'mask_id'
    """
    n = len(masks_data)
    adjacency = {i: 0 for i in range(n)}
    
    # Pre-dilate all masks for faster checking
    kernel = np.ones((dilate_pixels, dilate_pixels), np.uint8)
    dilated_masks = []
    bboxes = []
    
    for i in range(n):
        m = masks_data[i]["mask"]
        dilated = cv2.dilate(m, kernel, iterations=1)
        dilated_masks.append(dilated)
        
        # bounding box
        coords = cv2.findNonZero(m)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            bboxes.append((x - dilate_pixels, y - dilate_pixels, 
                           x + w + dilate_pixels, y + h + dilate_pixels))
        else:
            bboxes.append((0, 0, 0, 0))
            
    for i in range(n):
        xi1, yi1, xi2, yi2 = bboxes[i]
        for j in range(i + 1, n):
            xj1, yj1, xj2, yj2 = bboxes[j]
            # Check bbox intersection
            if not (xi2 < xj1 or xi1 > xj2 or yi2 < yj1 or yi1 > yj2):
                # Detailed check
                overlap = cv2.bitwise_and(dilated_masks[i], masks_data[j]["mask"])
                if cv2.countNonZero(overlap) > 0:
                    adjacency[i] += 1
                    adjacency[j] += 1
                    
    for i in range(n):
        masks_data[i]["adjacency"] = adjacency[i]
        
    return masks_data

def topological_analysis():
    project_root = Path(".")
    metadata_path = project_root / "data" / "metadata" / "dataset_metadata.json"
    mgr = MetadataManager(metadata_path)
    
    all_features = []
    
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
                if area > 10: # ignore absolute dust
                    masks_list.append({
                        "mask_id": mask_meta.mask_id,
                        "mask": mask,
                        "area": area
                    })
        
        # Compute adjacency
        masks_list = compute_adjacency(masks_list)
        
        # Sort by area for rank
        masks_list.sort(key=lambda x: x["area"], reverse=True)
        
        for rank, m in enumerate(masks_list):
            mask_img = m["mask"]
            # Extract basic geometric
            features = extract_geometric_features(mask_img, image_area=img_area)
            
            # Additional required features
            # Vertex Count
            contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                vertex_count = len(approx)
                x, y, w, h = cv2.boundingRect(c)
                bbox_area = w * h
            else:
                vertex_count = 0
                bbox_area = 0
                
            features.update({
                "image_id": image_id,
                "mask_id": m["mask_id"],
                "rank": rank + 1,
                "adjacency": m["adjacency"],
                "vertex_count": vertex_count,
                "bbox_area": bbox_area
            })
            
            # Determine if retained by previous logic
            passed_geom = (features["solidity"] >= 0.5) and (features["extent"] >= 0.4)
            passed_area = (features["relative_area"] >= 0.015)
            passed_rank = (rank + 1 <= 8)
            features["retained_by_old_filter"] = passed_geom and passed_area and passed_rank
            
            all_features.append(features)
            
    df = pd.DataFrame(all_features)
    df.to_csv("data/interim/topological_analysis.csv", index=False)
    
    print("--- Analysis Complete ---")
    print(f"Processed {len(df)} total valid masks.")
    retained = df[df['retained_by_old_filter'] == True]
    rejected = df[df['retained_by_old_filter'] == False]
    print(f"Old Filter Retained: {len(retained)}")
    print(f"Old Filter Rejected: {len(rejected)}")

if __name__ == "__main__":
    topological_analysis()
