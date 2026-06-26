import os
import json
from glob import glob

def main():
    print("--- AUDIT START ---")
    
    # 1. Audit YOLO labels for class IDs that might map to Artwork (like 6, 7 if mapped differently, or anything out of 0-5)
    yolo_label_dirs = [
        r"d:\Silvertouch Internship\task3\yolo_detection_dataset\train\labels",
        r"d:\Silvertouch Internship\task3\yolo_detection_dataset\valid\labels",
        r"d:\Silvertouch Internship\task3\yolo_detection_dataset\test\labels"
    ]
    
    yolo_counts = {}
    for d in yolo_label_dirs:
        if not os.path.exists(d): continue
        for txt in glob(os.path.join(d, "*.txt")):
            with open(txt, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cid = int(parts[0])
                        yolo_counts[cid] = yolo_counts.get(cid, 0) + 1
                        
    print("\nYOLO Label Distribution:")
    for k, v in sorted(yolo_counts.items()):
        print(f"Class ID {k}: {v} boxes")
        
    # 2. Audit COCO annotations (_annotations_mapped.json)
    coco_files = [
        r"d:\Silvertouch Internship\task3\train\_annotations_mapped.json",
        r"d:\Silvertouch Internship\task3\valid\_annotations_mapped.json",
        r"d:\Silvertouch Internship\task3\test\_annotations_mapped.json"
    ]
    
    coco_counts = {}
    for cfile in coco_files:
        if not os.path.exists(cfile): continue
        with open(cfile, "r", encoding="utf-8") as f:
            data = json.load(f)
            for ann in data.get("annotations", []):
                cid = ann.get("category_id")
                coco_counts[cid] = coco_counts.get(cid, 0) + 1
                
    print("\nCOCO _annotations_mapped.json Distribution:")
    for k, v in sorted(coco_counts.items()):
        print(f"Category ID {k}: {v} boxes")

if __name__ == "__main__":
    main()
