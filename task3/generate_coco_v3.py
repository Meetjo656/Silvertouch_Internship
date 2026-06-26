import os
import json
import numpy as np
from pathlib import Path
from collections import Counter
from PIL import Image
from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────────────
MODEL_PATH = r"d:\Silvertouch Internship\task3\trained_models\structural_8class_50epochs\weights\best.pt"
IMAGE_DIRS = {
    "Artwork": r"d:\Silvertouch Internship\task3\new (1)\images\Artwork",
    "Not_Artwork": r"d:\Silvertouch Internship\task3\new (1)\images\Not_Artwork",
}
OUTPUT_JSON = r"d:\Silvertouch Internship\task3\new (1)\annotations\annotations_v3.json"
REPORT_TXT = r"d:\Silvertouch Internship\task3\new (1)\annotations\dataset_report_v3.txt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# COCO categories
CATEGORIES = [
    {"id": 1, "name": "Artwork", "supercategory": "structural"},
    {"id": 2, "name": "not_artwork", "supercategory": "structural"},
    {"id": 3, "name": "Front_Panel", "supercategory": "structural"},
    {"id": 4, "name": "Side_Panel", "supercategory": "structural"},
    {"id": 5, "name": "Back_Panel", "supercategory": "structural"},
    {"id": 6, "name": "Extra_Areas", "supercategory": "structural"},
    {"id": 7, "name": "Bottom_Panel", "supercategory": "structural"},
    {"id": 8, "name": "Top_Panel", "supercategory": "structural"},
]

YOLO_TO_COCO = {i: i + 1 for i in range(8)}
COCO_TO_NAME = {c["id"]: c["name"] for c in CATEGORIES}

# Class-aware thresholds
CLASS_THRESHOLDS = {
    0: 0.10,  # Artwork
    1: 0.10,  # not_artwork
    2: 0.03,  # Front_Panel
    3: 0.03,  # Side_Panel
    4: 0.03,  # Back_Panel
    5: 0.15,  # Extra_Areas
    6: 0.03,  # Bottom_Panel
    7: 0.03,  # Top_Panel
}

def compute_overlap_stats(boxA, boxB):
    # box is [x_min, y_min, width, height]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    iou = 0.0
    union = boxAArea + boxBArea - interArea
    if union > 0:
        iou = interArea / float(union)
        
    return iou, interArea, boxAArea, boxBArea

def get_center(box):
    return box[0] + box[2] / 2.0, box[1] + box[3] / 2.0

def is_inside(cx, cy, box):
    return (box[0] <= cx <= box[0] + box[2]) and (box[1] <= cy <= box[1] + box[3])

def main():
    print(f"Loading model {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    coco = {
        "images": [],
        "annotations": [],
        "categories": CATEGORIES,
    }

    image_id = 0
    annotation_id = 0
    
    stats = {
        "total_images": 0,
        "images_with_annotations": 0,
        "total_annotations": 0,
        "iou_conflicts_detected": 0,
        "containment_conflicts_detected": 0,
        "center_conflicts_detected": 0,
        "artwork_removed": 0,
        "not_artwork_removed": 0,
        "front_panel_removed": 0,
        "back_panel_removed": 0,
        "top_panel_removed": 0,
        "bottom_panel_removed": 0,
        "nms_duplicates_removed": 0, # tracked internally by YOLO but we can note the setting
        "class_counts": Counter(),
        "confidences": {c["id"]: [] for c in CATEGORIES}
    }

    # Gather valid files
    all_files = {}
    for subdir_name, img_dir in IMAGE_DIRS.items():
        files = [f for f in os.listdir(img_dir) if Path(f).suffix.lower() in IMAGE_EXTENSIONS]
        files = [f for f in files if " - Copy" not in f]
        for f in files:
            all_files[f] = subdir_name

    print(f"Total unique images to process: {len(all_files)}")

    for subdir_name, img_dir in IMAGE_DIRS.items():
        image_files = sorted([f for f, d in all_files.items() if d == subdir_name])
        print(f"\nProcessing {subdir_name}: {len(image_files)} images")
        
        for i, img_file in enumerate(image_files, 1):
            img_path = os.path.join(img_dir, img_file)
            
            try:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"  [WARN] Skipping unreadable image {img_file}: {e}")
                continue

            image_id += 1
            coco["images"].append({
                "id": image_id,
                "file_name": img_file,
                "width": img_width,
                "height": img_height,
            })
            stats["total_images"] += 1

            # Run inference. 
            # conf=0.01 to get everything, we filter manually.
            # iou=0.50 to let YOLO handle class-wise NMS duplicates natively!
            results = model.predict(source=img_path, conf=0.01, iou=0.50, imgsz=640, verbose=False)
            boxes = results[0].boxes
            
            image_boxes = []
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    
                    if cls_id not in YOLO_TO_COCO:
                        continue
                        
                    # Class-aware threshold filter
                    if conf < CLASS_THRESHOLDS.get(cls_id, 0.25):
                        continue
                        
                    xyxy = box.xyxy[0].cpu().numpy()
                    x_min, y_min, x_max, y_max = xyxy
                    
                    # Clamp
                    x_min = max(0.0, float(x_min))
                    y_min = max(0.0, float(y_min))
                    x_max = min(float(img_width), float(x_max))
                    y_max = min(float(img_height), float(y_max))
                    
                    w_px = x_max - x_min
                    h_px = y_max - y_min
                    area = w_px * h_px
                    
                    if w_px <= 0 or h_px <= 0 or area <= 0:
                        continue
                        
                    cat_id = YOLO_TO_COCO[cls_id]
                    image_boxes.append({
                        "cat_id": cat_id,
                        "bbox": [x_min, y_min, w_px, h_px],
                        "area": area,
                        "conf": conf
                    })

            # --- Conflict Resolution: Artwork vs Not_Artwork ---
            to_remove = set()
            artwork_boxes = [(idx, b) for idx, b in enumerate(image_boxes) if b["cat_id"] == 1]
            not_artwork_boxes = [(idx, b) for idx, b in enumerate(image_boxes) if b["cat_id"] == 2]
            
            for idx_a, a_box in artwork_boxes:
                for idx_na, na_box in not_artwork_boxes:
                    b_a = a_box["bbox"]
                    b_na = na_box["bbox"]
                    
                    iou, inter_area, area_a, area_na = compute_overlap_stats(b_a, b_na)
                    containment_a = inter_area / area_a if area_a > 0 else 0
                    containment_b = inter_area / area_na if area_na > 0 else 0
                    
                    cx_a, cy_a = get_center(b_a)
                    cx_na, cy_na = get_center(b_na)
                    
                    is_iou_conflict = iou > 0.30
                    is_containment_conflict = (containment_a > 0.50) or (containment_b > 0.50)
                    is_center_conflict = is_inside(cx_na, cy_na, b_a) or is_inside(cx_a, cy_a, b_na)
                    
                    if is_iou_conflict or is_containment_conflict or is_center_conflict:
                        if is_center_conflict:
                            stats["center_conflicts_detected"] += 1
                        elif is_containment_conflict:
                            stats["containment_conflicts_detected"] += 1
                        elif is_iou_conflict:
                            stats["iou_conflicts_detected"] += 1
                            
                        # Case 1 & 2: Confidence wins
                        if a_box["conf"] > na_box["conf"]:
                            to_remove.add(idx_na)
                            stats["not_artwork_removed"] += 1
                        elif na_box["conf"] > a_box["conf"]:
                            to_remove.add(idx_a)
                            stats["artwork_removed"] += 1
                        else:
                            # Case 3: Larger box wins
                            if a_box["area"] > na_box["area"]:
                                to_remove.add(idx_na)
                                stats["not_artwork_removed"] += 1
                            else:
                                to_remove.add(idx_a)
                                stats["artwork_removed"] += 1

            # --- Structural Panel Top-1 Filtering ---
            TOP1_CLASSES = {
                3: "front_panel_removed",
                5: "back_panel_removed",
                7: "bottom_panel_removed",
                8: "top_panel_removed"
            }
            
            for cat_id, stat_key in TOP1_CLASSES.items():
                class_boxes = [(idx, b) for idx, b in enumerate(image_boxes) if b["cat_id"] == cat_id and idx not in to_remove]
                if len(class_boxes) > 1:
                    class_boxes.sort(key=lambda x: x[1]["conf"], reverse=True)
                    for idx, _ in class_boxes[1:]:
                        to_remove.add(idx)
                        stats[stat_key] += 1

            # --- Commit Valid Annotations ---
            img_has_annotations = False
            for idx, box in enumerate(image_boxes):
                if idx in to_remove:
                    continue
                    
                annotation_id += 1
                cat_id = box["cat_id"]
                
                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [
                        round(box["bbox"][0], 2),
                        round(box["bbox"][1], 2),
                        round(box["bbox"][2], 2),
                        round(box["bbox"][3], 2),
                    ],
                    "area": round(box["area"], 2),
                    "iscrowd": 0,
                })
                
                stats["total_annotations"] += 1
                stats["class_counts"][cat_id] += 1
                stats["confidences"][cat_id].append(box["conf"])
                img_has_annotations = True
                
            if img_has_annotations:
                stats["images_with_annotations"] += 1

            if i % 200 == 0 or i == len(image_files):
                print(f"  Processed {i}/{len(image_files)}")

    # ── Write Output & Generate Report ──────────────────────────────
    with open(OUTPUT_JSON, "w") as f:
        json.dump(coco, f, indent=2)

    report_lines = []
    report_lines.append("==================================================")
    report_lines.append("DATASET CONVERSION & DISTRIBUTION REPORT V3")
    report_lines.append("==================================================")
    report_lines.append(f"Total Images Analyzed : {stats['total_images']}")
    report_lines.append(f"Images w/ Annotations : {stats['images_with_annotations']}")
    report_lines.append(f"Images w/o Annotations: {stats['total_images'] - stats['images_with_annotations']}")
    report_lines.append(f"Total Annotations     : {stats['total_annotations']}")
    
    report_lines.append("\n--- Conflict Resolution Statistics ---")
    report_lines.append(f"IoU conflicts detected                : {stats['iou_conflicts_detected']}")
    report_lines.append(f"Containment conflicts detected        : {stats['containment_conflicts_detected']}")
    report_lines.append(f"Center-point conflicts detected       : {stats['center_conflicts_detected']}")
    report_lines.append(f"Total Artwork removals                : {stats['artwork_removed']}")
    report_lines.append(f"Total Not_Artwork removals            : {stats['not_artwork_removed']}")
    
    report_lines.append("\n--- Structural Panel Duplicate Filtering ---")
    report_lines.append(f"Front_Panel duplicates removed        : {stats['front_panel_removed']}")
    report_lines.append(f"Back_Panel duplicates removed         : {stats['back_panel_removed']}")
    report_lines.append(f"Top_Panel duplicates removed          : {stats['top_panel_removed']}")
    report_lines.append(f"Bottom_Panel duplicates removed       : {stats['bottom_panel_removed']}")

    report_lines.append("\n--- NMS Statistics ---")
    report_lines.append("Duplicate detections removed: Handled natively by YOLO (iou=0.50)")
    report_lines.append(f"Final detections retained   : {stats['total_annotations']}")

    report_lines.append("\n==================================================")
    report_lines.append("ANNOTATIONS BY CLASS (Total Box Count):")
    for cat in CATEGORIES:
        cid = cat["id"]
        cname = cat["name"]
        count = stats["class_counts"][cid]
        report_lines.append(f"  {cname:<15}: {count}")

    report_lines.append("\n==================================================")
    report_lines.append("CONFIDENCE STATISTICS (Retained Boxes):")
    for cat in CATEGORIES:
        cid = cat["id"]
        cname = cat["name"]
        confs = stats["confidences"][cid]
        if confs:
            c_min = np.min(confs)
            c_max = np.max(confs)
            c_mean = np.mean(confs)
            c_med = np.median(confs)
            report_lines.append(f"  {cname:<15} -> Min: {c_min:.3f} | Max: {c_max:.3f} | Mean: {c_mean:.3f} | Median: {c_med:.3f}")
        else:
            report_lines.append(f"  {cname:<15} -> No detections")
            
    report_lines.append("==================================================")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    
    with open(REPORT_TXT, "w") as f:
        f.write(report_text + "\n")
        
    print(f"\nSUCCESS: Saved CVAT-Ready COCO JSON -> {OUTPUT_JSON}")
    print(f"SUCCESS: Saved Dataset Statistics   -> {REPORT_TXT}")

if __name__ == "__main__":
    main()
