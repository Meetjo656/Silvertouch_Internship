import os
import json
from pathlib import Path
from ultralytics import YOLO
import numpy as np

# Configuration
JSON_PATH = r"d:\Silvertouch Internship\task3\new (1)\annotations\annotations_v2.json"
MODEL_PATH = r"d:\Silvertouch Internship\task3\trained_models\structural_8class_50epochs\weights\best.pt"
IMG_BASE_DIRS = {
    "Artwork": r"d:\Silvertouch Internship\task3\new (1)\images\Artwork",
    "Not_Artwork": r"d:\Silvertouch Internship\task3\new (1)\images\Not_Artwork"
}

# COCO categories from the json
# 1: Artwork, 2: not_artwork, 3: Front_Panel, 4: Side_Panel, 
# 5: Back_Panel, 6: Extra_Areas, 7: Bottom_Panel, 8: Top_Panel
PANEL_CLASSES = {3, 4, 5, 7, 8}
EXTRA_AREAS_CLASS = 6

def find_target_images():
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
        
    img_by_id = {img["id"]: img for img in data["images"]}
    annotations_by_img = {img["id"]: [] for img in data["images"]}
    
    for ann in data["annotations"]:
        annotations_by_img[ann["image_id"]].append(ann)
        
    target_images = []
    
    # We are looking for images that have Extra_Areas (6) but zero PANEL classes (3,4,5,7,8)
    for img_id, anns in annotations_by_img.items():
        classes_present = set(a["category_id"] for a in anns)
        if EXTRA_AREAS_CLASS in classes_present and len(classes_present.intersection(PANEL_CLASSES)) == 0:
            target_images.append(img_by_id[img_id])
            if len(target_images) == 100:
                break
                
    return target_images

def resolve_path(filename):
    for subdir, path in IMG_BASE_DIRS.items():
        if " - Copy" in filename: continue
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            return full_path
    return None

def main():
    print("Finding 100 target images...")
    target_images = find_target_images()
    print(f"Found {len(target_images)} target images.")
    
    print(f"Loading model {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    report_file = open(r"d:\Silvertouch Internship\task3\raw_confidence_audit.txt", "w")
    report_file.write("RAW CONFIDENCE AUDIT REPORT\n")
    report_file.write("===========================\n\n")
    
    class_names = {
        0: "Artwork", 1: "not_artwork", 2: "Front_Panel", 3: "Side_Panel",
        4: "Back_Panel", 5: "Extra_Areas", 6: "Bottom_Panel", 7: "Top_Panel"
    }
    
    for img_meta in target_images:
        filename = img_meta["file_name"]
        img_path = resolve_path(filename)
        if not img_path: continue
        
        report_file.write(f"Image: {filename}\n")
        
        # Run raw inference with very low confidence
        # Disable Agnostic NMS, but we want to see before standard NMS ideally.
        # Ultralytics doesn't easily expose PRE-NMS boxes. We can set iou=0.99 to basically disable NMS.
        results = model.predict(source=img_path, conf=0.01, iou=0.99, verbose=False, imgsz=640)
        
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            report_file.write("  No detections even at conf=0.01\n\n")
            continue
            
        # Group highest confidence per class for this image
        max_conf_per_class = {}
        all_detections = []
        
        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            all_detections.append((cls_id, conf))
            if cls_id not in max_conf_per_class or conf > max_conf_per_class[cls_id]:
                max_conf_per_class[cls_id] = conf
                
        # Write max confidences
        for cls_id in sorted(max_conf_per_class.keys()):
            name = class_names[cls_id]
            report_file.write(f"  {name:<15}: {max_conf_per_class[cls_id]:.4f}\n")
            
        # Check if panels were found but below 0.15 threshold
        panels_found_low_conf = False
        panels_found_high_conf = False
        for cls_id, conf in max_conf_per_class.items():
            if cls_id in [2, 3, 4, 6, 7]: # YOLO 0-indexed panel classes
                if conf < 0.15:
                    panels_found_low_conf = True
                else:
                    panels_found_high_conf = True
                    
        if panels_found_low_conf and not panels_found_high_conf:
            report_file.write("  -> CONCLUSION: Panels exist but confidence is below 0.15!\n")
        elif not panels_found_low_conf and not panels_found_high_conf:
            report_file.write("  -> CONCLUSION: Model absolutely failed to generate panel boxes even at conf=0.01\n")
        elif panels_found_high_conf:
            report_file.write("  -> CONCLUSION: Panel has conf >= 0.15, but was suppressed by NMS from Extra_Areas!\n")
            
        report_file.write("\n")
        
    report_file.close()
    print("Audit complete. Results written to raw_confidence_audit.txt")

if __name__ == '__main__':
    main()
