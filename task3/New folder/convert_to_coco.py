"""
Convert YOLO inference results to COCO JSON format for CVAT import.
Reads all images from Artwork and Not_Artwork, uses the YOLO txt annotations,
and produces a single COCO-format annotations JSON.
"""

import os
import json
from pathlib import Path
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

OUTPUT_PATH = r"d:\Silvertouch Internship\task3\new (1)\annotations\annotations.json"

# COCO categories (1-indexed for COCO standard)
CATEGORIES = [
    {"id": 1, "name": "Artwork", "supercategory": "structural"},
    {"id": 2, "name": "Not_Artwork", "supercategory": "structural"},
    {"id": 3, "name": "Front_Panel", "supercategory": "structural"},
    {"id": 4, "name": "Side_Panel", "supercategory": "structural"},
    {"id": 5, "name": "Back_Panel", "supercategory": "structural"},
    {"id": 6, "name": "Extra_Areas", "supercategory": "structural"},
    {"id": 7, "name": "Bottom_Part", "supercategory": "structural"},
    {"id": 8, "name": "Top_Panel", "supercategory": "structural"},
]

# YOLO class_id (0-indexed) -> COCO category_id (1-indexed)
YOLO_TO_COCO = {i: i + 1 for i in range(8)}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def convert():
    coco = {
        "images": [],
        "annotations": [],
        "categories": CATEGORIES,
    }

    image_id = 0
    annotation_id = 0
    total_images = 0
    total_annotations = 0

    for subdir_name, img_dir in IMAGE_DIRS.items():
        ann_dir = ANNOTATION_DIRS[subdir_name]
        print(f"\nProcessing: {subdir_name}")
        print(f"  Images dir : {img_dir}")
        print(f"  Annot. dir : {ann_dir}")

        image_files = sorted([
            f for f in os.listdir(img_dir)
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS
            and " - Copy" not in f  # skip duplicates not in CVAT
        ])
        print(f"  Found {len(image_files)} images (excluded ' - Copy' duplicates)")

        for i, img_file in enumerate(image_files, 1):
            img_path = os.path.join(img_dir, img_file)
            stem = Path(img_file).stem

            # Get actual image dimensions
            try:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"  ⚠ Skipping {img_file}: {e}")
                continue

            image_id += 1

            # file_name is just the image filename (CVAT flat upload)
            coco["images"].append({
                "id": image_id,
                "file_name": img_file,
                "width": img_width,
                "height": img_height,
            })

            # Read corresponding YOLO txt annotation
            txt_path = os.path.join(ann_dir, f"{stem}.txt")
            if not os.path.exists(txt_path):
                if i % 100 == 0:
                    print(f"  [{i}/{len(image_files)}] {img_file} (no annotation file)")
                continue

            with open(txt_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                yolo_cls = int(parts[0])
                x_center_norm = float(parts[1])
                y_center_norm = float(parts[2])
                w_norm = float(parts[3])
                h_norm = float(parts[4])
                conf = float(parts[5]) if len(parts) > 5 else 1.0

                # Convert normalised xywh (center) -> COCO bbox [x_min, y_min, w, h] in pixels
                w_px = w_norm * img_width
                h_px = h_norm * img_height
                x_min = (x_center_norm * img_width) - (w_px / 2)
                y_min = (y_center_norm * img_height) - (h_px / 2)

                # Clamp to image boundaries
                x_min = max(0.0, x_min)
                y_min = max(0.0, y_min)
                w_px = min(w_px, img_width - x_min)
                h_px = min(h_px, img_height - y_min)

                area = w_px * h_px

                annotation_id += 1
                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": YOLO_TO_COCO[yolo_cls],
                    "bbox": [
                        round(x_min, 2),
                        round(y_min, 2),
                        round(w_px, 2),
                        round(h_px, 2),
                    ],
                    "area": round(area, 2),
                    "iscrowd": 0,
                })
                total_annotations += 1

            total_images += 1

            if i % 100 == 0 or i == len(image_files):
                print(f"  [{i}/{len(image_files)}] Processed")

    # Write COCO JSON
    with open(OUTPUT_PATH, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"\n{'='*60}")
    print(f"COCO ANNOTATION EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"  Images     : {len(coco['images'])}")
    print(f"  Annotations: {len(coco['annotations'])}")
    print(f"  Categories : {len(coco['categories'])}")
    print(f"  Output     : {OUTPUT_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    convert()
