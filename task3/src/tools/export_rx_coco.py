import json
import os
import cv2

def main():
    base_dir = r"d:\Silvertouch Internship\task3"
    results_path = os.path.join(base_dir, "rx_detection_results.json")
    out_coco_path = os.path.join(base_dir, "rx_annotations_coco.json")
    
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
        
    categories = [
        {"id": 1, "name": "Rx"}
    ]
    
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    img_id_counter = 1
    ann_id_counter = 1
    
    for item in results:
        # We only need to export annotations for images where Rx was found,
        # but for completeness in CVAT we could export all.
        # Let's export all so the dataset matches.
        img_name = item["image_name"]
        panel_type = item["panel_type"]
        
        # Determine actual file path to get width and height
        # Original image was in: front_back_clean/{panel_type}/filename
        # Wait, the img_name stored in JSON is `f"{panel_type}_{filename}"`.
        # So we need to reconstruct the original filename, which is img_name without the prefix.
        prefix = f"{panel_type}_"
        if img_name.startswith(prefix):
            original_filename = img_name[len(prefix):]
        else:
            original_filename = img_name
            
        img_path = os.path.join(base_dir, "front_back_clean", panel_type, original_filename)
        
        # Read image to get dims
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        height, width = img.shape[:2]
        
        image_id = img_id_counter
        coco_data["images"].append({
            "id": image_id,
            "file_name": img_name, # Storing the unique output filename as used in overlays
            "width": width,
            "height": height
        })
        
        for match in item.get("rx_matches", []):
            poly = match["bbox"] # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            
            xs = [pt[0] for pt in poly]
            ys = [pt[1] for pt in poly]
            
            x_min = min(xs)
            y_min = min(ys)
            w = max(xs) - x_min
            h = max(ys) - y_min
            
            area = w * h
            
            coco_data["annotations"].append({
                "id": ann_id_counter,
                "image_id": image_id,
                "category_id": 1, # Rx
                "bbox": [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)],
                "area": round(area, 2),
                "iscrowd": 0,
                "confidence": match["confidence"]
            })
            
            ann_id_counter += 1
            
        img_id_counter += 1

    with open(out_coco_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=4)
        
    print(f"Exported {ann_id_counter - 1} annotations across {img_id_counter - 1} images.")
    print(f"Saved to {out_coco_path}")

if __name__ == "__main__":
    main()
