"""
Run the structural_8class_50epochs YOLO model on Artwork and Not_Artwork
image directories and save annotations in YOLO format + a JSON summary.
"""

import os
import json
from pathlib import Path
from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────────────
MODEL_PATH = r"d:\Silvertouch Internship\task3\trained_models\ankesh_sir model\latest.pth"

IMAGE_DIRS = [
    r"d:\Silvertouch Internship\task3\new (1)\images\Artwork",
    r"d:\Silvertouch Internship\task3\new (1)\images\Not_Artwork",
]

OUTPUT_BASE = r"d:\Silvertouch Internship\task3\new (1)\annotations"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7
IMG_SIZE = 512  # same as training

CLASS_NAMES = {
    0: "Artwork",
    1: "Not_Artwork",
    2: "Front_Panel",
    3: "Side_Panel",
    4: "Back_Panel",
    5: "Extra_Areas",
    6: "Bottom_Part",
    7: "Top_Panel",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def run_inference():
    # Load model
    print(f"Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully!\n")

    all_results_json = {}
    total_images = 0
    total_detections = 0

    for img_dir in IMAGE_DIRS:
        dir_name = Path(img_dir).name  # "Artwork" or "Not_Artwork"
        print(f"{'='*60}")
        print(f"Processing directory: {dir_name}")
        print(f"Path: {img_dir}")
        print(f"{'='*60}")

        # Create output directories
        annotation_dir = os.path.join(OUTPUT_BASE, dir_name)
        os.makedirs(annotation_dir, exist_ok=True)

        # Collect image files
        image_files = sorted([
            f for f in os.listdir(img_dir)
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS
        ])
        print(f"Found {len(image_files)} images\n")

        dir_results = []

        for i, img_file in enumerate(image_files, 1):
            img_path = os.path.join(img_dir, img_file)
            stem = Path(img_file).stem

            # Run inference
            results = model.predict(
                source=img_path,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=IMG_SIZE,
                verbose=False,
            )

            result = results[0]
            boxes = result.boxes

            # Save YOLO-format annotation (class x_center y_center width height confidence)
            txt_path = os.path.join(annotation_dir, f"{stem}.txt")
            img_detections = []

            with open(txt_path, "w") as f:
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls_id = int(box.cls.item())
                        conf = float(box.conf.item())
                        # xywhn = normalized x_center, y_center, width, height
                        xywhn = box.xywhn[0].tolist()
                        x_center, y_center, w, h = xywhn

                        f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {conf:.4f}\n")

                        img_detections.append({
                            "class_id": cls_id,
                            "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                            "confidence": round(conf, 4),
                            "bbox_normalized": {
                                "x_center": round(x_center, 6),
                                "y_center": round(y_center, 6),
                                "width": round(w, 6),
                                "height": round(h, 6),
                            },
                        })

            num_det = len(img_detections)
            total_images += 1
            total_detections += num_det

            dir_results.append({
                "image": img_file,
                "num_detections": num_det,
                "detections": img_detections,
            })

            if i % 50 == 0 or i == len(image_files):
                print(f"  [{i}/{len(image_files)}] Processed - {img_file} ({num_det} detections)")

        all_results_json[dir_name] = dir_results
        print(f"\n  ✓ Saved {len(image_files)} annotation files to: {annotation_dir}\n")

    # Save consolidated JSON report
    json_output_path = os.path.join(OUTPUT_BASE, "inference_results.json")
    summary = {
        "model": MODEL_PATH,
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "img_size": IMG_SIZE,
        "total_images": total_images,
        "total_detections": total_detections,
        "class_names": CLASS_NAMES,
        "results_by_directory": all_results_json,
    }

    with open(json_output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{'='*60}")
    print(f"INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"Total images processed : {total_images}")
    print(f"Total detections       : {total_detections}")
    print(f"YOLO txt annotations   : {OUTPUT_BASE}/<subdir>/*.txt")
    print(f"JSON summary           : {json_output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_inference()
