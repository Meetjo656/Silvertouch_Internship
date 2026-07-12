import os
import cv2
import json
import shutil
import statistics
from pathlib import Path

def main():
    base_dir = Path("d:/Silvertouch Internship/task3")
    images_dir = base_dir / "Only Images"
    
    cleaned_dir = base_dir / "dataset_cleaned" / "images"
    v2_dir = base_dir / "dataset_v2" / "images"
    
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    v2_dir.mkdir(parents=True, exist_ok=True)
    
    invalid_images = []
    widths = []
    heights = []
    
    images_before = 0
    images_removed = 0
    corrupted_images = 0
    blank_images = 0 # Not filtering these, but keeping counter for report compatibility if needed, though they asked not to. Let's set to 0.
    
    print("Starting Phase 1 & 2: Validation and Resolution Audit...")
    
    valid_image_paths = []
    
    for filename in os.listdir(images_dir):
        # We assume only files are in this directory
        file_path = images_dir / filename
        if not file_path.is_file():
            continue
            
        images_before += 1
        
        # Check zero-byte
        if os.path.getsize(file_path) == 0:
            invalid_images.append({
                "filename": filename,
                "reason": "zero-byte file"
            })
            corrupted_images += 1
            images_removed += 1
            continue
            
        # Try to load with OpenCV
        try:
            # imread handles various formats. Sometimes it returns None if corrupted or unsupported.
            img = cv2.imread(str(file_path))
            if img is None:
                invalid_images.append({
                    "filename": filename,
                    "reason": "OpenCV cannot load image / unsupported format"
                })
                corrupted_images += 1
                images_removed += 1
                continue
                
            # If loaded successfully, record dimensions
            h, w, _ = img.shape
            widths.append(w)
            heights.append(h)
            
            valid_image_paths.append(file_path)
            
        except Exception as e:
            invalid_images.append({
                "filename": filename,
                "reason": f"Exception during load: {str(e)}"
            })
            corrupted_images += 1
            images_removed += 1
    
    print(f"Validation complete. Valid images: {len(valid_image_paths)}, Invalid: {len(invalid_images)}")
    
    # Generate invalid_images.txt
    invalid_txt_path = base_dir / "invalid_images.txt"
    with open(invalid_txt_path, "w", encoding="utf-8") as f:
        for item in invalid_images:
            f.write(f"{item['filename']} - {item['reason']}\n")
            
    # Generate resolution_audit.json
    print("Generating resolution audit...")
    resolution_audit = {
        "width": {
            "min": min(widths) if widths else 0,
            "median": statistics.median(widths) if widths else 0,
            "max": max(widths) if widths else 0
        },
        "height": {
            "min": min(heights) if heights else 0,
            "median": statistics.median(heights) if heights else 0,
            "max": max(heights) if heights else 0
        }
    }
    with open(base_dir / "resolution_audit.json", "w") as f:
        json.dump(resolution_audit, f, indent=4)
        
    print("Copying valid images to dataset_cleaned and dataset_v2...")
    for file_path in valid_image_paths:
        shutil.copy2(file_path, cleaned_dir / file_path.name)
        shutil.copy2(file_path, v2_dir / file_path.name)
        
    images_after = len(valid_image_paths)
    
    dataset_cleaning_report = {
        "images_before": images_before,
        "images_after": images_after,
        "images_removed": images_removed,
        "corrupted_images": corrupted_images,
        "blank_images": 0,
        "resize_failures": 0,
        "final_dataset_size": images_after
    }
    
    with open(base_dir / "dataset_cleaning_report.json", "w") as f:
        json.dump(dataset_cleaning_report, f, indent=4)
        
    print("Computing class distribution from predicted_annotations_mapped.json...")
    annotations_path = base_dir / "predicted_annotations_mapped.json"
    class_distribution_report = {}
    
    if annotations_path.exists():
        with open(annotations_path, "r") as f:
            data = json.load(f)
            
        categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
        class_counts = {name: 0 for name in categories.values()}
        
        for ann in data.get("annotations", []):
            cat_id = ann.get("category_id")
            if cat_id in categories:
                class_counts[categories[cat_id]] += 1
                
        class_distribution_report = class_counts
        
        with open(base_dir / "class_distribution_report.json", "w") as f:
            json.dump(class_distribution_report, f, indent=4)
    else:
        print("Warning: predicted_annotations_mapped.json not found.")
        
    print("All tasks completed successfully!")

if __name__ == "__main__":
    main()
