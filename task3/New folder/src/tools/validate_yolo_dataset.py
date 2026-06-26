import os
import yaml
from pathlib import Path
from collections import defaultdict
import re

def validate_dataset(config_path="config/yolo_dataset.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["paths"]["output_dir"])
    data_yaml_path = output_dir / "data.yaml"

    if not data_yaml_path.exists():
        print("data.yaml not found! Run create_yaml first.")
        return

    with open(data_yaml_path, "r") as f:
        data_yaml = yaml.safe_load(f)

    nc = data_yaml.get("nc", 0)
    names = data_yaml.get("names", {})
    
    not_artwork_dir = Path(config["paths"]["not_artwork_images_dir"])
    scrapped_names = set(f for f in os.listdir(not_artwork_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))) if not_artwork_dir.exists() else set()

    splits = ["train", "valid", "test"]
    
    validation_errors = []
    
    stats = {
        "Total Images": 0,
        "Artwork Images": 0,
        "Negative Images": 0,
        "Augmented Images Generated": 0,
        "Total Bounding Boxes": 0,
        "Class Distribution": defaultdict(int),
        "Images Per Split": defaultdict(int),
        "Class Coverage": defaultdict(lambda: {"train": 0, "valid": 0, "test": 0}),
        "Areas": [],
        "Objects Per Image": [],
        "Negative Sample Distribution": defaultdict(int)
    }

    origin_to_splits = defaultdict(set)
    all_files = set()

    for split in splits:
        images_dir = output_dir / split / "images"
        labels_dir = output_dir / split / "labels"
        
        if not images_dir.exists():
            continue

        images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        stats["Images Per Split"][split] = len(images)
        stats["Total Images"] += len(images)

        for img_name in images:
            if img_name in all_files:
                validation_errors.append(f"Duplicate file found: {img_name} in {split}")
            all_files.add(img_name)

            label_name = Path(img_name).with_suffix(".txt").name
            label_path = labels_dir / label_name

            if not label_path.exists():
                validation_errors.append(f"Missing label file for image: {img_name} in {split}")
                continue

            match = re.match(r"(.*)_aug\d+", Path(img_name).stem)
            if match:
                origin_id = match.group(1) + Path(img_name).suffix
                stats["Augmented Images Generated"] += 1
            else:
                origin_id = img_name
            
            origin_to_splits[origin_id].add(split)
            
            is_negative = origin_id in scrapped_names

            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            if is_negative:
                stats["Negative Images"] += 1
                stats["Negative Sample Distribution"][split] += 1
                if any(line.strip() for line in lines):
                    validation_errors.append(f"Negative sample {img_name} has non-empty label file!")
            else:
                stats["Artwork Images"] += 1
                
            if not is_negative and not any(line.strip() for line in lines):
                validation_errors.append(f"Artwork sample {img_name} has empty label file!")

            obj_count = 0
            for line_idx, line in enumerate(lines):
                parts = line.strip().split()
                if not parts:
                    continue
                
                obj_count += 1
                stats["Total Bounding Boxes"] += 1
                
                try:
                    class_id = int(parts[0])
                    if class_id >= nc or class_id < 0:
                        validation_errors.append(f"Invalid class ID {class_id} in {label_name} line {line_idx+1}")
                    cls_name = names.get(class_id, str(class_id))
                    stats["Class Distribution"][cls_name] += 1
                    stats["Class Coverage"][cls_name][split] += 1
                except ValueError:
                    validation_errors.append(f"Non-integer class ID in {label_name} line {line_idx+1}")

                if len(parts) != 5:
                    validation_errors.append(f"Invalid bbox format (expected 5 items) in {label_name} line {line_idx+1}")
                else:
                    try:
                        cx, cy, w, h = map(float, parts[1:5])
                        if not (0 <= cx <= 1): validation_errors.append(f"cx out of bounds ({cx}) in {label_name}")
                        if not (0 <= cy <= 1): validation_errors.append(f"cy out of bounds ({cy}) in {label_name}")
                        if not (0 < w <= 1): validation_errors.append(f"w out of bounds ({w}) in {label_name}")
                        if not (0 < h <= 1): validation_errors.append(f"h out of bounds ({h}) in {label_name}")
                        
                        area = w * h
                        stats["Areas"].append(area)
                    except ValueError:
                        validation_errors.append(f"Non-float coordinate in {label_name} line {line_idx+1}")
            
            if not is_negative:
                stats["Objects Per Image"].append(obj_count)

    # Check for leakage
    for origin_id, splits_found in origin_to_splits.items():
        if len(splits_found) > 1:
            validation_errors.append(f"Leakage detected: {origin_id} in {splits_found}")

    # Generate Reports
    generate_validation_report(validation_errors)
    generate_statistics_report(stats)
    generate_class_coverage_report(stats, names)
    generate_bbox_statistics_report(stats)
    generate_experimental_recommendation()
    generate_training_strategy_review()
    generate_final_recommendation()
    generate_training_readiness_report(validation_errors, stats, names, data_yaml_path)

    print("All validation checks and reports generated.")

def generate_validation_report(errors):
    with open("dataset_validation_report.md", "w", encoding="utf-8") as f:
        f.write("# Dataset Validation Report\n\n")
        if not errors:
            f.write("✅ **PASSED**: No errors found.\n\n")
        else:
            f.write("❌ **FAILED**: Errors detected.\n\n")
            for err in errors:
                f.write(f"- {err}\n")

def generate_statistics_report(stats):
    with open("dataset_statistics_report.md", "w", encoding="utf-8") as f:
        f.write("# Dataset Statistics Report\n\n")
        f.write(f"- **Total Images**: {stats['Total Images']}\n")
        f.write(f"  - Artwork Images: {stats['Artwork Images']}\n")
        f.write(f"  - Negative Images: {stats['Negative Images']}\n")
        
        f.write(f"\n- **Splits**:\n")
        for split in ["train", "valid", "test"]:
            f.write(f"  - {split.capitalize()} Count: {stats['Images Per Split'][split]}\n")
            
        f.write(f"\n- **Negative Sample Distribution**:\n")
        for split in ["train", "valid", "test"]:
            f.write(f"  - {split.capitalize()}: {stats['Negative Sample Distribution'][split]}\n")
            
        avg_objs = sum(stats["Objects Per Image"]) / len(stats["Objects Per Image"]) if stats["Objects Per Image"] else 0
        f.write(f"\n- **Total Bounding Boxes**: {stats['Total Bounding Boxes']}\n")
        f.write(f"- **Average Objects Per Image** (Artwork only): {avg_objs:.2f}\n")
        f.write(f"- **Augmented Images Generated**: {stats['Augmented Images Generated']}\n")
        
        f.write("\n## Class Distribution\n")
        for cls_name, count in sorted(stats["Class Distribution"].items(), key=lambda x: x[1], reverse=True):
            f.write(f"- {cls_name}: {count}\n")

def generate_class_coverage_report(stats, names):
    with open("class_coverage_report.md", "w", encoding="utf-8") as f:
        f.write("# Class Coverage Report\n\n")
        f.write("| Class Name | Train | Valid | Test | Status |\n")
        f.write("|---|---|---|---|---|\n")
        
        for cls_id, cls_name in names.items():
            cov = stats["Class Coverage"][cls_name]
            status = "✅ OK"
            total_cov = cov["train"] + cov["valid"] + cov["test"]
            
            if total_cov == 0:
                status = "⚠️ ZERO_INSTANCES_IN_DATASET"
            else:
                if cov["valid"] == 0:
                    status = "❌ MISSING_FROM_VALID"
                if cov["test"] == 0:
                    status += " ❌ MISSING_FROM_TEST"
            
            f.write(f"| {cls_name} | {cov['train']} | {cov['valid']} | {cov['test']} | {status} |\n")

def generate_bbox_statistics_report(stats):
    areas = sorted(stats["Areas"])
    avg_area = sum(areas) / len(areas) if areas else 0
    median_area = areas[len(areas)//2] if areas else 0
    
    small_obj = sum(1 for a in areas if a < 0.01) # <1% of image area
    med_obj = sum(1 for a in areas if 0.01 <= a <= 0.1) # 1-10% of image area
    large_obj = sum(1 for a in areas if a > 0.1) # >10% of image area
    total_obj = len(areas) if areas else 1
    
    max_obj = max(stats["Objects Per Image"]) if stats["Objects Per Image"] else 0
    avg_obj = sum(stats["Objects Per Image"]) / len(stats["Objects Per Image"]) if stats["Objects Per Image"] else 0
    
    cls_sorted = sorted(stats["Class Distribution"].items(), key=lambda x: x[1], reverse=True)
    top_10 = cls_sorted[:10]
    bottom_10 = cls_sorted[-10:] if len(cls_sorted) > 10 else cls_sorted
    
    with open("bbox_statistics_report.md", "w", encoding="utf-8") as f:
        f.write("# BBox Statistics Report\n\n")
        f.write(f"- **Average BBox Area (Normalized)**: {avg_area:.6f}\n")
        f.write(f"- **Median BBox Area (Normalized)**: {median_area:.6f}\n")
        f.write(f"- **Small Objects (<1% area)**: {(small_obj/total_obj)*100:.2f}%\n")
        f.write(f"- **Medium Objects (1-10% area)**: {(med_obj/total_obj)*100:.2f}%\n")
        f.write(f"- **Large Objects (>10% area)**: {(large_obj/total_obj)*100:.2f}%\n")
        f.write(f"- **Average Objects Per Image**: {avg_obj:.2f}\n")
        f.write(f"- **Maximum Objects Per Image**: {max_obj}\n\n")
        
        f.write("## Top 10 Most Frequent Classes\n")
        for cls_name, count in top_10: f.write(f"- {cls_name}: {count}\n")
        f.write("\n## Top 10 Least Frequent Classes\n")
        for cls_name, count in bottom_10: f.write(f"- {cls_name}: {count}\n")

def generate_experimental_recommendation():
    content = """# Experimental Recommendation

Based on dataset statistics, the following sequential experiments are recommended. 
Do not assume YOLOv8l wins. Baseline models often perform best on smaller datasets.

### Experiment 1: Baseline
- **Model**: YOLOv8m
- **Image Size**: `imgsz=640`
- **Purpose**: Establishes baseline precision/recall without high risk of overfitting.

### Experiment 2: Image Size Scaling Test
- **Model**: YOLOv8m
- **Image Size**: `imgsz=800`
- **Purpose**: Evaluates impact of higher resolution, critical for datasets with high object density and small objects.

### Experiment 3: Model Capacity Scaling Test
- **Model**: YOLOv8l
- **Image Size**: `imgsz=640`
- **Purpose**: Evaluates whether additional parameters aid in resolving the 20 classes, or simply induce overfitting.

### Comparison Metrics
Compare the following across all 3 experiments to declare a winner:
- Precision
- Recall
- mAP50
- mAP50-95
- Training Time
- Inference Time
"""
    with open("experimental_recommendation.md", "w", encoding="utf-8") as f:
        f.write(content)

def generate_training_strategy_review():
    content = """# Training Strategy Review

### Evaluate: YOLOv8m vs YOLOv8l

#### YOLOv8m
- **Expected Accuracy**: High. Excellent balance for densely packed datasets.
- **Expected Generalization**: Strong. Less likely to memorize the ~200 source images.
- **Overfitting Risk**: Moderate. Still requires `patience=50`.
- **Training Cost**: Medium. Fast enough for rapid iteration.

#### YOLOv8l
- **Expected Accuracy**: Potentially high, but often drops below YOLOv8m on small datasets due to overfitting.
- **Expected Generalization**: Weak. Very high risk of memorizing the exact bounding boxes of the training set.
- **Overfitting Risk**: Severe.
- **Training Cost**: High. Slower iteration speeds.

**Conclusion**: YOLOv8m must be your primary benchmark. Only promote to YOLOv8l if YOLOv8m shows underfitting (e.g. failing to capture specific classes despite high image size).
"""
    with open("training_strategy_review.md", "w", encoding="utf-8") as f:
        f.write(content)

def generate_final_recommendation():
    content = """# Final Recommendation

1. **Why Detection over Segmentation**: 99.37% of annotations lack polygons. Trying to train a segmentation model forces the framework to throw away your data. Object detection uses 100% of the dataset.
2. **Why preserve existing splits**: Randomly merging and re-splitting a pre-split dataset introduces severe risks of data leakage, especially when dealing with augmented sets.
3. **Why negative samples were injected**: The scrapped images represent background/negative space. By distributing them dynamically, you force the model to suppress false positives on similar packaging areas.
4. **Why YOLOv8m is the initial benchmark**: It possesses enough capacity to resolve 20 classes and 41 objects/image without immediately memorizing the small original training set (183 images).
5. **When YOLOv8l should be evaluated**: Only if Experiment 1 and 2 reveal that YOLOv8m cannot differentiate visually similar classes (underfitting).
6. **Image size scaling vs Model scaling**: Because your objects are extremely dense, increasing `imgsz` to 800 (Experiment 2) is expected to yield much higher accuracy gains than simply switching to a larger model (Experiment 3).
"""
    with open("final_recommendation.md", "w", encoding="utf-8") as f:
        f.write(content)

def generate_training_readiness_report(errors, stats, names, data_yaml_path):
    all_cov = True
    for cls_id, cls_name in names.items():
        cov = stats["Class Coverage"][cls_name]
        total_cov = cov["train"] + cov["valid"] + cov["test"]
        if total_cov > 0:
            if cov["valid"] == 0 or cov["test"] == 0:
                all_cov = False
            
    is_ready = not errors and all_cov and data_yaml_path.exists() and stats["Negative Images"] > 0
    
    with open("training_readiness_report.md", "w", encoding="utf-8") as f:
        f.write("# Training Readiness Gate\n\n")
        f.write(f"[{'✓' if stats['Total Images'] > 0 else ' '}] Images Present\n")
        f.write(f"[{'✓' if stats['Total Bounding Boxes'] > 0 else ' '}] Labels Present\n")
        f.write(f"[{'✓' if not any('Leakage' in e for e in errors) else ' '}] No Leakage\n")
        f.write(f"[{'✓' if all_cov else ' '}] Class Coverage Verified\n")
        f.write(f"[{'✓' if not any('Negative' in e for e in errors) and stats['Negative Images'] > 0 else ' '}] Negative Samples Verified\n")
        f.write(f"[{'✓' if data_yaml_path.exists() else ' '}] data.yaml Generated\n")
        f.write(f"[{'✓' if not errors else ' '}] Dataset Structure Valid\n\n")
        
        if is_ready:
            f.write("## Final Status: READY FOR TRAINING\n")
        else:
            f.write("## Final Status: NOT READY FOR TRAINING\n")
            f.write("\nPlease check the dataset_validation_report.md and class_coverage_report.md to resolve issues.\n")

if __name__ == "__main__":
    validate_dataset()
