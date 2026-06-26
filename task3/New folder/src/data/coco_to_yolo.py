import json
import os
import shutil
from pathlib import Path
import yaml

def convert_coco_to_yolo(config_path="config/yolo_dataset.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    splits = ["train", "valid", "test"]
    output_dir = Path(paths["output_dir"])

    # Prepare category mapping from train set
    with open(paths["train_coco_json"], "r", encoding="utf-8") as f:
        train_coco = json.load(f)
    
    valid_categories = [c for c in train_coco["categories"] if c["name"] not in ["Artwork", "not_artwork"]]
    
    category_map = {
        category["id"]: idx
        for idx, category in enumerate(valid_categories)
    }

    # Generate category mapping report
    report_lines = ["# Category Mapping Report\n", "| Original COCO ID | YOLO Class ID | Class Name |", "|---|---|---|"]
    for category in valid_categories:
        idx = category_map[category["id"]]
        report_lines.append(f"| {category['id']} | {idx} | {category['name']} |")
    
    with open("category_mapping_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # Keep track of artwork image counts to dynamically inject negative samples later
    artwork_counts = {"train": 0, "valid": 0, "test": 0}

    # 1. Process Artwork Images
    for split in splits:
        coco_json_path = paths[f"{split}_coco_json"]
        images_dir = Path(paths[f"{split}_images_dir"])
        
        split_images_out = output_dir / split / "images"
        split_labels_out = output_dir / split / "labels"
        split_images_out.mkdir(parents=True, exist_ok=True)
        split_labels_out.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(coco_json_path):
            print(f"Warning: Missing json for {split}")
            continue

        with open(coco_json_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        annotations_by_image = {}
        for anno in coco_data.get("annotations", []):
            image_id = anno["image_id"]
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(anno)

        for image in coco_data.get("images", []):
            image_id = image["id"]
            file_name = image["file_name"]
            width = image["width"]
            height = image["height"]

            src_image_path = images_dir / file_name
            if not src_image_path.exists():
                src_image_path = Path(coco_json_path).parent / file_name
            
            if src_image_path.exists():
                dst_image_path = split_images_out / file_name
                shutil.copy2(src_image_path, dst_image_path)
                artwork_counts[split] += 1
            else:
                print(f"Warning: Artwork image not found {src_image_path}")
                continue

            labels_content = []
            annos = annotations_by_image.get(image_id, [])
            for anno in annos:
                if "bbox" not in anno or anno["category_id"] not in category_map:
                    continue
                
                yolo_class_id = category_map[anno["category_id"]]
                x, y, w, h = anno["bbox"]
                
                # Convert COCO bbox [top_left_x, top_left_y, width, height] to YOLO normalized [center_x, center_y, width, height]
                cx = (x + (w / 2.0)) / width
                cy = (y + (h / 2.0)) / height
                nw = w / width
                nh = h / height
                
                # Clamp between 0 and 1
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.000001, min(1.0, nw))
                nh = max(0.000001, min(1.0, nh))
                
                labels_content.append(f"{yolo_class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            label_file_name = Path(file_name).with_suffix(".txt").name
            label_dst_path = split_labels_out / label_file_name
            with open(label_dst_path, "w", encoding="utf-8") as f:
                f.write("\n".join(labels_content))

    # 2. Process Negative Samples (Scrapped)
    not_artwork_dir = Path(paths["not_artwork_images_dir"])
    if not_artwork_dir.exists():
        total_artwork = sum(artwork_counts.values())
        if total_artwork > 0:
            scrapped_images = [f for f in sorted(os.listdir(not_artwork_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total_scrapped = len(scrapped_images)
            
            # Dynamic proportional splitting
            train_ratio = artwork_counts["train"] / total_artwork
            valid_ratio = artwork_counts["valid"] / total_artwork
            test_ratio = artwork_counts["test"] / total_artwork
            
            train_neg_count = int(round(total_scrapped * train_ratio))
            valid_neg_count = int(round(total_scrapped * valid_ratio))
            # Test gets remainder
            test_neg_count = total_scrapped - train_neg_count - valid_neg_count
            
            split_allocations = (
                ["train"] * train_neg_count +
                ["valid"] * valid_neg_count +
                ["test"] * test_neg_count
            )
            
            for file_name, split in zip(scrapped_images, split_allocations):
                src_path = not_artwork_dir / file_name
                split_images_out = output_dir / split / "images"
                split_labels_out = output_dir / split / "labels"
                
                dst_image_path = split_images_out / file_name
                shutil.copy2(src_path, dst_image_path)
                
                label_file_name = Path(file_name).with_suffix(".txt").name
                label_dst_path = split_labels_out / label_file_name
                with open(label_dst_path, "w", encoding="utf-8") as f:
                    f.write("")  # empty label file
    else:
        print(f"Warning: not_artwork_images_dir not found {not_artwork_dir}")

if __name__ == "__main__":
    convert_coco_to_yolo()
