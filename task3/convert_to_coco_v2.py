"""
Convert YOLO inference results to COCO JSON format for CVAT import.
Includes strict filename validation, dataset analytics, and confidence filtering
for high-recall pre-annotation workflows.
"""

import os
import sys
import json
from pathlib import Path
from collections import Counter
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────────
IMAGE_DIRS = {
    "Artwork": r"d:\Silvertouch Internship\task3\new (1)\images\Artwork",
    "Not_Artwork": r"d:\Silvertouch Internship\task3\new (1)\images\Not_Artwork",
}

ANNOTATION_DIRS = {
    "Artwork": r"d:\Silvertouch Internship\task3\new (1)\annotations\Artwork",
    "Not_Artwork": r"d:\Silvertouch Internship\task3\new (1)\annotations\Not_Artwork",
}

OUTPUT_JSON = r"d:\Silvertouch Internship\task3\new (1)\annotations\annotations_v2.json"
REPORT_TXT = r"d:\Silvertouch Internship\task3\new (1)\annotations\dataset_report.txt"

# High Recall Threshold for Pre-Annotation
CONF_THRESHOLD = 0.15

# COCO categories (1-indexed for COCO standard)
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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    if interArea == 0.0:
        return 0.0

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def convert():
    print(f"Starting conversion with CONF_THRESHOLD = {CONF_THRESHOLD}")
    
    # ── 1. Filename Validation ─────────────────────────────────────────
    all_files = {}
    duplicate_collisions = []
    
    for subdir_name, img_dir in IMAGE_DIRS.items():
        files = [f for f in os.listdir(img_dir) if Path(f).suffix.lower() in IMAGE_EXTENSIONS]
        # Ignore Windows "- Copy" duplicates as they won't be uploaded to CVAT
        files = [f for f in files if " - Copy" not in f]
        
        for f in files:
            if f in all_files:
                duplicate_collisions.append(f)
            else:
                all_files[f] = subdir_name

    if duplicate_collisions:
        print("\n[ERROR] CRITICAL: Duplicate filenames detected across directories!")
        print("CVAT flat import will overwrite these images and invalidate your annotations.")
        for d in duplicate_collisions[:10]:
            print(f"  - {d}")
        if len(duplicate_collisions) > 10:
            print(f"  ... and {len(duplicate_collisions) - 10} more.")
        print("\nPlease rename these files before importing into CVAT.")
        sys.exit(1)
        
    print(f"SUCCESS: Filename validation passed. Total unique images: {len(all_files)}")

    # ── 2. Conversion and Analytics ────────────────────────────────────
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
        "filtered_by_conf": 0,
        "corrupted_lines": 0,
        "conflicts_detected": 0,
        "conflicts_removed": 0,
        "class_counts": Counter(),
        "class_image_counts": Counter()
    }

    for subdir_name, img_dir in IMAGE_DIRS.items():
        ann_dir = ANNOTATION_DIRS[subdir_name]
        print(f"\nProcessing: {subdir_name}")
        
        # Get images mapping to this subdir
        image_files = sorted([f for f, d in all_files.items() if d == subdir_name])
        
        for i, img_file in enumerate(image_files, 1):
            img_path = os.path.join(img_dir, img_file)
            stem = Path(img_file).stem

            try:
                with Image.open(img_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    if img.size != (640, 640):
                        img = img.resize((640, 640), Image.Resampling.LANCZOS)
                        img.save(img_path)
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
            
            txt_path = os.path.join(ann_dir, f"{stem}.txt")
            if not os.path.exists(txt_path):
                continue

            with open(txt_path, "r") as f:
                lines = f.readlines()

            img_classes_present = set()
            img_has_valid_box = False
            
            image_boxes = []
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    stats["corrupted_lines"] += 1
                    continue
                
                try:
                    yolo_cls = int(parts[0])
                    x_center_norm = float(parts[1])
                    y_center_norm = float(parts[2])
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])
                    conf = float(parts[5]) if len(parts) > 5 else 1.0
                except ValueError:
                    stats["corrupted_lines"] += 1
                    continue
                
                if yolo_cls not in YOLO_TO_COCO:
                    stats["corrupted_lines"] += 1
                    continue

                if conf < CONF_THRESHOLD:
                    stats["filtered_by_conf"] += 1
                    continue

                # Convert to absolute COCO format
                w_px = w_norm * img_width
                h_px = h_norm * img_height
                x_min = (x_center_norm * img_width) - (w_px / 2)
                y_min = (y_center_norm * img_height) - (h_px / 2)

                # Clamp
                x_min = max(0.0, x_min)
                y_min = max(0.0, y_min)
                w_px = min(w_px, img_width - x_min)
                h_px = min(h_px, img_height - y_min)

                area = w_px * h_px
                if area <= 0:
                    continue

                cat_id = YOLO_TO_COCO[yolo_cls]
                image_boxes.append({
                    "cat_id": cat_id,
                    "bbox": [x_min, y_min, w_px, h_px],
                    "area": area,
                    "conf": conf
                })

            # --- Conflict Resolution ---
            to_remove = set()
            artwork_boxes = [(i, b) for i, b in enumerate(image_boxes) if b["cat_id"] == 1]
            not_artwork_boxes = [(i, b) for i, b in enumerate(image_boxes) if b["cat_id"] == 2]
            
            for i, a_box in artwork_boxes:
                for j, na_box in not_artwork_boxes:
                    iou = compute_iou(a_box["bbox"], na_box["bbox"])
                    if iou > 0.50:
                        stats["conflicts_detected"] += 1
                        if a_box["conf"] > na_box["conf"]:
                            to_remove.add(j)
                        elif na_box["conf"] > a_box["conf"]:
                            to_remove.add(i)
                        else:
                            if a_box["area"] > na_box["area"]:
                                to_remove.add(j)
                            else:
                                to_remove.add(i)
                                
            stats["conflicts_removed"] += len(to_remove)

            # Append surviving boxes
            for idx, box in enumerate(image_boxes):
                if idx in to_remove:
                    continue
                    
                annotation_id += 1
                
                # CVAT prefers purely structural COCO format (no extra fields like score)
                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": box["cat_id"],
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
                stats["class_counts"][box["cat_id"]] += 1
                img_classes_present.add(box["cat_id"])
                img_has_valid_box = True

            if img_has_valid_box:
                stats["images_with_annotations"] += 1
                for c in img_classes_present:
                    stats["class_image_counts"][c] += 1

            if i % 200 == 0 or i == len(image_files):
                print(f"  [{i}/{len(image_files)}] processed")

    # ── 3. Write Output & Generate Report ──────────────────────────────
    with open(OUTPUT_JSON, "w") as f:
        json.dump(coco, f, indent=2)

    report_lines = []
    report_lines.append("==================================================")
    report_lines.append("DATASET CONVERSION & DISTRIBUTION REPORT")
    report_lines.append("==================================================")
    report_lines.append(f"Confidence Threshold  : {CONF_THRESHOLD}")
    report_lines.append(f"Total Images Analyzed : {stats['total_images']}")
    report_lines.append(f"Images w/ Annotations : {stats['images_with_annotations']}")
    report_lines.append(f"Images w/o Annotations: {stats['total_images'] - stats['images_with_annotations']}")
    report_lines.append(f"Total Annotations     : {stats['total_annotations']}")
    
    avg_boxes = stats['total_annotations'] / max(1, stats['images_with_annotations'])
    report_lines.append(f"Avg Boxes per Image   : {avg_boxes:.2f} (for images with boxes)")
    report_lines.append(f"Filtered by Conf      : {stats['filtered_by_conf']}")
    report_lines.append(f"Corrupted Lines       : {stats['corrupted_lines']}")
    report_lines.append(f"Conflicts Detected    : {stats['conflicts_detected']}")
    report_lines.append(f"Conflicts Removed     : {stats['conflicts_removed']}")
    report_lines.append("==================================================")
    report_lines.append("ANNOTATIONS BY CLASS (Total Box Count):")
    for cat in CATEGORIES:
        cid = cat["id"]
        cname = cat["name"]
        count = stats["class_counts"][cid]
        pct = (count / max(1, stats['total_annotations'])) * 100
        report_lines.append(f"  {cname:<15}: {count:<6} ({pct:>5.1f}%)")
        
    report_lines.append("--------------------------------------------------")
    report_lines.append("IMAGES CONTAINING CLASS:")
    for cat in CATEGORIES:
        cid = cat["id"]
        cname = cat["name"]
        count = stats["class_image_counts"][cid]
        pct = (count / max(1, stats['total_images'])) * 100
        report_lines.append(f"  {cname:<15}: {count:<6} ({pct:>5.1f}%)")
    report_lines.append("==================================================")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    
    with open(REPORT_TXT, "w") as f:
        f.write(report_text + "\n")
        
    print(f"\nSUCCESS: Saved COCO JSON    -> {OUTPUT_JSON}")
    print(f"SUCCESS: Saved Audit Report -> {REPORT_TXT}")


if __name__ == "__main__":
    convert()
