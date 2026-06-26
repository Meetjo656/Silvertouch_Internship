import os
import json
from pathlib import Path
from ultralytics import YOLO

JSON_PATH = r"d:\Silvertouch Internship\task3\new (1)\annotations\annotations_v3.json"
MODEL_PATH = r"d:\Silvertouch Internship\task3\trained_models\structural_8class_50epochs\weights\best.pt"
IMG_BASE_DIRS = {
    "Artwork": r"d:\Silvertouch Internship\task3\new (1)\images\Artwork",
    "Not_Artwork": r"d:\Silvertouch Internship\task3\new (1)\images\Not_Artwork"
}

# COCO Categories
# 3: Front_Panel, 4: Side_Panel, 5: Back_Panel
# YOLO classes
# 2: Front_Panel, 3: Side_Panel, 4: Back_Panel

CONF_THRESHOLD = 0.15

def resolve_path(filename):
    for subdir, path in IMG_BASE_DIRS.items():
        if " - Copy" in filename: continue
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            return full_path
    return None

def main():
    print(f"Loading annotations from {JSON_PATH}...")
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
        
    img_by_id = {img["id"]: img["file_name"] for img in data["images"]}
    annotations_by_img = {img["id"]: set() for img in data["images"]}
    for ann in data["annotations"]:
        annotations_by_img[ann["image_id"]].add(ann["category_id"])
        
    print(f"Loading YOLO model {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    # Metrics to track
    visible_front = 0
    visible_front_missing_ann = 0
    
    visible_back = 0
    visible_back_missing_ann = 0
    
    visible_side = 0
    visible_side_missing_ann = 0
    
    processed_count = 0
    
    # Gather paths
    image_paths_to_run = []
    img_id_order = []
    
    for img_id, filename in img_by_id.items():
        img_path = resolve_path(filename)
        if img_path:
            image_paths_to_run.append(img_path)
            img_id_order.append(img_id)
            
    print(f"Found {len(image_paths_to_run)} images on disk. Running inference...")
    
    # Process in batches to speed up
    batch_size = 32
    for i in range(0, len(image_paths_to_run), batch_size):
        batch_paths = image_paths_to_run[i:i+batch_size]
        batch_ids = img_id_order[i:i+batch_size]
        
        results = model.predict(source=batch_paths, conf=CONF_THRESHOLD, iou=0.8, verbose=False, imgsz=640)
        
        for j, res in enumerate(results):
            img_id = batch_ids[j]
            anns = annotations_by_img[img_id]
            
            boxes = res.boxes
            detected_classes = set()
            if boxes is not None:
                for box in boxes:
                    detected_classes.add(int(box.cls.item()))
                    
            # Check Front (YOLO cls 2, COCO cat 3)
            if 2 in detected_classes:
                visible_front += 1
                if 3 not in anns:
                    visible_front_missing_ann += 1
                    
            # Check Back (YOLO cls 4, COCO cat 5)
            if 4 in detected_classes:
                visible_back += 1
                if 5 not in anns:
                    visible_back_missing_ann += 1
                    
            # Check Side (YOLO cls 3, COCO cat 4)
            if 3 in detected_classes:
                visible_side += 1
                if 4 not in anns:
                    visible_side_missing_ann += 1
                    
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count}/{len(image_paths_to_run)} images...")
                
    print("\n--- Visibility vs Annotation Analysis ---")
    print(f"Total images analyzed: {processed_count}")
    print(f"\nFront Panel:")
    print(f"  Visible (predicted by model) : {visible_front}")
    print(f"  Visible but Annotation Missing: {visible_front_missing_ann} ({visible_front_missing_ann/max(1, visible_front)*100:.2f}% of visible)")
    
    print(f"\nBack Panel:")
    print(f"  Visible (predicted by model) : {visible_back}")
    print(f"  Visible but Annotation Missing: {visible_back_missing_ann} ({visible_back_missing_ann/max(1, visible_back)*100:.2f}% of visible)")
    
    print(f"\nSide Panel:")
    print(f"  Visible (predicted by model) : {visible_side}")
    print(f"  Visible but Annotation Missing: {visible_side_missing_ann} ({visible_side_missing_ann/max(1, visible_side)*100:.2f}% of visible)")
    
if __name__ == '__main__':
    main()
