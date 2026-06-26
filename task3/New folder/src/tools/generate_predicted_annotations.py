import os
import cv2
import json
import numpy as np
from ultralytics import YOLO

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def validate_coco_data(coco_data, class_names):
    # Validation logic
    valid = True
    errors = []
    
    # Check 1: Verify every annotation references a valid image_id
    valid_image_ids = {img["id"] for img in coco_data["images"]}
    img_id_errors = 0
    
    # Check 2: Verify category IDs
    valid_category_ids = {cat["id"] for cat in coco_data["categories"]}
    cat_id_errors = 0
    
    # Check 3: Verify bounds
    bounds_errors = 0
    
    # Check 4: Unique Annotation IDs
    ann_ids = set()
    duplicate_ann_ids = 0
    
    # Image dimension map for bounds checking
    img_dims = {img["id"]: {"w": img["width"], "h": img["height"]} for img in coco_data["images"]}
    
    for ann in coco_data["annotations"]:
        if ann["id"] in ann_ids:
            duplicate_ann_ids += 1
        ann_ids.add(ann["id"])
        
        if ann["image_id"] not in valid_image_ids:
            img_id_errors += 1
            
        if ann["category_id"] not in valid_category_ids:
            cat_id_errors += 1
            
        # Bounds check
        x, y, w, h = ann["bbox"]
        img_w = img_dims.get(ann["image_id"], {}).get("w", 0)
        img_h = img_dims.get(ann["image_id"], {}).get("h", 0)
        
        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            bounds_errors += 1
            
    if img_id_errors > 0: errors.append(f"{img_id_errors} annotations have invalid image_ids.")
    if cat_id_errors > 0: errors.append(f"{cat_id_errors} annotations have invalid category_ids.")
    if bounds_errors > 0: errors.append(f"{bounds_errors} annotations have out-of-bounds bounding boxes.")
    if duplicate_ann_ids > 0: errors.append(f"{duplicate_ann_ids} annotations have duplicate IDs.")
    
    if len(errors) > 0:
        print("VALIDATION FAILED:")
        for e in errors:
            print(" -", e)
        return False
    
    print("VALIDATION PASSED: All annotation references, categories, bounds, and IDs are valid.")
    return True


def main():
    base_dir = r"d:\Silvertouch Internship\task3"
    input_dir = os.path.join(base_dir, "Only Images")
    model_path = os.path.join(base_dir, r"trained_models\structural_8class_30epochs\weights\best.pt")
    
    out_json_path = os.path.join(base_dir, "predicted_annotations_mapped.json")
    report_json_path = os.path.join(base_dir, "predicted_annotation_report.json")
    
    CONF_THRESHOLD = 0.25
    print(f"Using Confidence Threshold: {CONF_THRESHOLD}")
    
    print("Loading YOLO model...")
    model = YOLO(model_path)
    # Ensure correct mapping
    categories = [
        {"id": 1, "name": "Artwork"},
        {"id": 2, "name": "not_artwork"},
        {"id": 3, "name": "Front_Panel"},
        {"id": 4, "name": "Side_Panel"},
        {"id": 5, "name": "Back_Panel"},
        {"id": 6, "name": "Extra_Areas"},
        {"id": 7, "name": "Bottom_Panel"},
        {"id": 8, "name": "Top_Panel"}
    ]
    
    # YOLO returns a dictionary for names, we'll assume it aligns but explicitly use the categories mapping for the ID
    cat_name_to_id = {cat["name"]: cat["id"] for cat in categories}
    yolo_id_to_name = model.names
    
    all_images = []
    if os.path.exists(input_dir):
        for f in os.listdir(input_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(input_dir, f))
                
    print(f"Found {len(all_images)} images for prediction.")
    
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    report_data = {
        "images_processed": 0,
        "total_annotations": 0,
        "class_counts": {cat["name"]: 0 for cat in categories},
        "average_confidence_per_class": {cat["name"]: [] for cat in categories}
    }
    
    ann_id_counter = 1
    
    for idx, img_path in enumerate(all_images):
        filename = os.path.basename(img_path)
        print(f"[{idx+1}/{len(all_images)}] Processing: {filename}")
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        height, width = img.shape[:2]
        
        image_id = idx + 1
        
        coco_data["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })
        
        results = model(img, verbose=False)
        report_data["images_processed"] += 1
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = round(float(box.conf[0].item()), 4)
                
                if conf < CONF_THRESHOLD:
                    continue
                    
                yolo_cls_id = int(box.cls[0].item())
                cls_name = yolo_id_to_name[yolo_cls_id]
                
                # Remap YOLO's Bottom_Part to CVAT's expected Bottom_Panel
                if cls_name == "Bottom_Part":
                    cls_name = "Bottom_Panel"
                elif cls_name == "Not_Artwork":
                    cls_name = "not_artwork"
                
                # Defensive lookup in case mapping differs slightly
                if cls_name not in cat_name_to_id:
                    continue
                    
                cat_id = cat_name_to_id[cls_name]
                
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                
                # Clip bounds
                x1 = max(0.0, x1)
                y1 = max(0.0, y1)
                x2 = min(float(width), x2)
                y2 = min(float(height), y2)
                
                w = x2 - x1
                h = y2 - y1
                
                if w <= 0 or h <= 0:
                    continue
                
                area = w * h
                
                coco_data["annotations"].append({
                    "id": ann_id_counter,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "area": round(area, 2),
                    "iscrowd": 0,
                    "confidence": conf
                })
                
                ann_id_counter += 1
                
                report_data["total_annotations"] += 1
                report_data["class_counts"][cls_name] += 1
                report_data["average_confidence_per_class"][cls_name].append(conf)

    # Calculate average confidences
    for cls_name, confs in report_data["average_confidence_per_class"].items():
        if len(confs) > 0:
            report_data["average_confidence_per_class"][cls_name] = round(float(np.mean(confs)), 4)
        else:
            report_data["average_confidence_per_class"][cls_name] = 0.0

    print("Running COCO Validation...")
    is_valid = validate_coco_data(coco_data, [c["name"] for c in categories])
    report_data["coco_validation_passed"] = is_valid
    
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=4)
        
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=4)
        
    print(f"Successfully generated pseudo-annotations!")
    print(f"Saved to {out_json_path}")
    print(f"Report saved to {report_json_path}")

if __name__ == "__main__":
    main()
