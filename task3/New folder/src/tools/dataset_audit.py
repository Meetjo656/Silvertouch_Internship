import os
import cv2
import json
import statistics
from pathlib import Path

def main():
    base_dir = Path("d:/Silvertouch Internship/task3")
    dataset_dir = base_dir / "yolo_detection_dataset"
    
    classes = {
        0: "Artwork",
        1: "Not_Artwork",
        2: "Front_Panel",
        3: "Side_Panel",
        4: "Back_Panel",
        5: "Extra_Areas",
        6: "Bottom_Panel",
        7: "Top_Panel"
    }
    
    # We will gather stats for each class
    class_counts = {name: 0 for name in classes.values()}
    class_bboxes = {name: [] for name in classes.values()}
    
    splits = ["train", "valid", "test"]
    
    for split in splits:
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        
        if not labels_dir.exists() or not images_dir.exists():
            continue
            
        for label_file in labels_dir.iterdir():
            if label_file.suffix != '.txt':
                continue
                
            image_file = images_dir / (label_file.stem + ".jpg")
            if not image_file.exists():
                image_file = images_dir / (label_file.stem + ".png")
                
            if not image_file.exists():
                continue
                
            # Read image to get dimensions
            img = cv2.imread(str(image_file))
            if img is None:
                continue
            
            img_h, img_w, _ = img.shape
            
            with open(label_file, "r") as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                    
                class_id = int(parts[0])
                if class_id not in classes:
                    continue
                    
                norm_w = float(parts[3])
                norm_h = float(parts[4])
                
                abs_w = norm_w * img_w
                abs_h = norm_h * img_h
                area = abs_w * abs_h
                
                class_name = classes[class_id]
                class_counts[class_name] += 1
                class_bboxes[class_name].append({
                    "width": abs_w,
                    "height": abs_h,
                    "area": area
                })
                
    # Filter only the requested panels for the bbox distribution report
    target_panels = ["Front_Panel", "Back_Panel", "Side_Panel", "Top_Panel", "Bottom_Panel", "Extra_Areas"]
    
    # Generate class_balance_report.json
    total_annotations = sum(class_counts.values())
    class_balance_report = {
        "total_annotations": total_annotations,
        "class_counts": class_counts,
        "relative_frequency": {k: (v / total_annotations if total_annotations > 0 else 0) for k, v in class_counts.items()}
    }
    
    with open(base_dir / "class_balance_report.json", "w") as f:
        json.dump(class_balance_report, f, indent=4)
        
    # Generate bbox_distribution_report.json
    bbox_report = {}
    for class_name in target_panels:
        bboxes = class_bboxes[class_name]
        if len(bboxes) == 0:
            bbox_report[class_name] = {
                "median_width": 0, "median_height": 0,
                "median_area": 0, "min_area": 0, "max_area": 0, "count": 0
            }
            continue
            
        widths = [b["width"] for b in bboxes]
        heights = [b["height"] for b in bboxes]
        areas = [b["area"] for b in bboxes]
        
        bbox_report[class_name] = {
            "count": len(bboxes),
            "median_width": statistics.median(widths),
            "median_height": statistics.median(heights),
            "median_area": statistics.median(areas),
            "min_area": min(areas),
            "max_area": max(areas)
        }
        
    with open(base_dir / "bbox_distribution_report.json", "w") as f:
        json.dump(bbox_report, f, indent=4)
        
    print("Dataset audit complete. Reports generated.")

if __name__ == "__main__":
    main()
