import os
import glob
import json
import hashlib
from datetime import datetime
from typing import Any
from PIL import Image
import shutil

YOLO_DIR = r"d:\Silvertouch Internship\task3\final_training_package\dataset"
OUTPUT_DIR = r"d:\Silvertouch Internship\task3\rfdetr_training_pipeline\dataset"
DOCS_DIR = r"d:\Silvertouch Internship\task3\rfdetr_training_pipeline\documentation"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

CLASSES = {
    0: 'Artwork',
    1: 'not_artwork',
    2: 'Front_Panel',
    3: 'Side_Panel',
    4: 'Back_Panel',
    5: 'Extra_Areas',
    6: 'Bottom_Panel',
    7: 'Top_Panel'
}

SPLITS = ['train', 'valid', 'test']

def get_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    print("Starting YOLO to COCO conversion...")
    
    total_images = 0
    total_annotations = 0
    class_distribution = {c: 0 for c in CLASSES.values()}
    
    report_lines = ["# YOLO to COCO Conversion Report\n", f"**Generated:** {datetime.now().isoformat()}\n"]
    manifest: dict[str, Any] = {
        "dataset_version": "1.0",
        "creation_timestamp": datetime.now().isoformat(),
        "source_dataset_path": YOLO_DIR,
        "number_of_classes": len(CLASSES),
        "generator_version": "1.0"
    }
    
    overall_largest = (0, 0)
    overall_smallest = (float('inf'), float('inf'))
    images_without_annotations = 0
    
    split_image_counts = {}
    
    for split in SPLITS:
        print(f"Processing split: {split}")
        split_img_dir = os.path.join(YOLO_DIR, split, "images")
        split_lbl_dir = os.path.join(YOLO_DIR, split, "labels")
        
        out_split_dir = os.path.join(OUTPUT_DIR, split)
        out_img_dir = os.path.join(out_split_dir, "images")
        os.makedirs(out_img_dir, exist_ok=True)
        
        coco_data = {
            "info": {
                "description": "Converted from YOLO Production Dataset",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [],
            "categories": [{"id": k, "name": v, "supercategory": "none"} for k, v in CLASSES.items()],
            "images": [],
            "annotations": []
        }
        
        if not os.path.exists(split_img_dir):
            print(f"Warning: Missing image directory {split_img_dir}")
            continue
            
        images = [f for f in os.listdir(split_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        split_image_counts[split] = len(images)
        
        img_id_counter = 1
        ann_id_counter = 1
        
        split_class_dist = {c: 0 for c in CLASSES.values()}
        split_missing_labels = []
        split_empty_labels = []
        
        for img_name in images:
            img_path = os.path.join(split_img_dir, img_name)
            
            # Copy image to new dataset structure
            dest_img_path = os.path.join(out_img_dir, img_name)
            if not os.path.exists(dest_img_path):
                shutil.copy2(img_path, dest_img_path)
            
            try:
                with Image.open(img_path) as im:
                    width, height = im.size
            except Exception as e:
                print(f"Failed to read image {img_path}: {e}")
                continue
                
            if width * height > overall_largest[0] * overall_largest[1]:
                overall_largest = (width, height)
            if width * height < overall_smallest[0] * overall_smallest[1]:
                overall_smallest = (width, height)
                
            img_info = {
                "id": img_id_counter,
                "file_name": img_name,
                "width": width,
                "height": height
            }
            coco_data["images"].append(img_info)
            total_images += 1
            
            base_name = os.path.splitext(img_name)[0]
            lbl_path = os.path.join(split_lbl_dir, base_name + ".txt")
            
            has_annotations = False
            
            if os.path.exists(lbl_path):
                with open(lbl_path, "r") as f:
                    lines = f.readlines()
                    if not lines:
                        split_empty_labels.append(img_name)
                        images_without_annotations += 1
                        
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            x_center, y_center, w, h = map(float, parts[1:5])
                            
                            # Convert YOLO relative to absolute COCO
                            abs_w = w * width
                            abs_h = h * height
                            abs_x_min = (x_center * width) - (abs_w / 2)
                            abs_y_min = (y_center * height) - (abs_h / 2)
                            
                            # Clip to image bounds
                            abs_x_min = max(0, abs_x_min)
                            abs_y_min = max(0, abs_y_min)
                            abs_w = min(width - abs_x_min, abs_w)
                            abs_h = min(height - abs_y_min, abs_h)
                            
                            ann_info = {
                                "id": ann_id_counter,
                                "image_id": img_id_counter,
                                "category_id": cls_id,
                                "bbox": [abs_x_min, abs_y_min, abs_w, abs_h],
                                "area": abs_w * abs_h,
                                "segmentation": [],
                                "iscrowd": 0
                            }
                            coco_data["annotations"].append(ann_info)
                            
                            if cls_id in CLASSES:
                                class_name = CLASSES[cls_id]
                                class_distribution[class_name] += 1
                                split_class_dist[class_name] += 1
                                
                            ann_id_counter += 1
                            total_annotations += 1
                            has_annotations = True
            else:
                split_missing_labels.append(img_name)
                images_without_annotations += 1
                
            img_id_counter += 1
            
        json_path = os.path.join(out_split_dir, "annotations.json")
        with open(json_path, "w") as f:
            json.dump(coco_data, f, indent=2)
            
        # Write to report
        report_lines.append(f"## Split: {split.upper()}\n")
        report_lines.append(f"- **Images Converted:** {len(images)}\n")
        report_lines.append(f"- **Annotations Converted:** {ann_id_counter - 1}\n")
        report_lines.append(f"- **Images missing label files:** {len(split_missing_labels)}\n")
        report_lines.append(f"- **Images with empty label files:** {len(split_empty_labels)}\n")
        report_lines.append("\n**Class Distribution:**\n")
        for c, count in split_class_dist.items():
            report_lines.append(f"- {c}: {count}\n")
        report_lines.append("\n")
        
        # Manifest updates
        manifest[f"{split}_images"] = len(images)
        manifest[f"{split}_annotations_sha256"] = get_sha256(json_path)
        
        # Create image list and hash it
        list_path = os.path.join(out_split_dir, "image_list.txt")
        with open(list_path, "w") as f:
            f.write("\n".join(sorted(images)))
        manifest[f"{split}_image_list_sha256"] = get_sha256(list_path)

    avg_instances = total_annotations / total_images if total_images > 0 else 0
    
    report_lines.insert(2, f"## Overall Summary\n")
    report_lines.insert(3, f"- **Total Images:** {total_images}\n")
    report_lines.insert(4, f"- **Total Annotations:** {total_annotations}\n")
    report_lines.insert(5, f"- **Average Instances per Image:** {avg_instances:.2f}\n")
    report_lines.insert(6, f"- **Images without Annotations:** {images_without_annotations}\n")
    report_lines.insert(7, f"- **Largest Image:** {overall_largest[0]}x{overall_largest[1]}\n")
    report_lines.insert(8, f"- **Smallest Image:** {overall_smallest[0]}x{overall_smallest[1]}\n\n")
    
    report_lines.insert(9, "**Overall Class Distribution:**\n")
    idx = 10
    for c, count in class_distribution.items():
        report_lines.insert(idx, f"- {c}: {count}\n")
        idx += 1
    report_lines.insert(idx, "\n")
    
    with open(os.path.join(DOCS_DIR, "conversion_report.md"), "w") as f:
        f.writelines(report_lines)
        
    manifest["number_of_images"] = total_images
    manifest["number_of_annotations"] = total_annotations
    manifest["class_distribution"] = class_distribution
    
    with open(os.path.join(OUTPUT_DIR, "dataset_manifest.yaml"), "w") as f:
        yaml_str = "dataset_version: '1.0'\n"
        for k, v in manifest.items():
            if k == "dataset_version": continue
            yaml_str += f"{k}: {v}\n"
        f.write(yaml_str)

    print("Conversion complete!")
    print(f"Report saved to {os.path.join(DOCS_DIR, 'conversion_report.md')}")
    print(f"Manifest saved to {os.path.join(OUTPUT_DIR, 'dataset_manifest.yaml')}")

if __name__ == "__main__":
    main()
