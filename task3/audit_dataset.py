import os
import json
import random

random.seed(42) # For reproducible output

splits = {
    "train": {
        "images": "d:/Silvertouch Internship/task3/final_training/train (1)/images/Train",
        "annotations": "d:/Silvertouch Internship/task3/final_training/train (1)/annotations/instances_Train.json"
    },
    "val": {
        "images": "d:/Silvertouch Internship/task3/final_training/Validation (1)/images/Train",
        "annotations": "d:/Silvertouch Internship/task3/final_training/Validation (1)/annotations/instances_Train.json"
    },
    "test": {
        "images": "d:/Silvertouch Internship/task3/final_training/test (1)/images/Train",
        "annotations": "d:/Silvertouch Internship/task3/final_training/test (1)/annotations/instances_Train.json"
    }
}

missing_on_disk_all = []
missing_in_json_all = []

for split, paths in splits.items():
    print(f"==================================================")
    print(f"SPLIT: {split.upper()}")
    
    img_dir = os.path.abspath(paths["images"])
    ann_path = os.path.abspath(paths["annotations"])
    print(f"Image Directory: {img_dir}")
    print(f"Annotation JSON: {ann_path}")
    
    if not os.path.exists(img_dir):
        print(f"ERROR: Image dir not found: {img_dir}")
        continue
    if not os.path.exists(ann_path):
        print(f"ERROR: Annotation file not found: {ann_path}")
        continue
        
    actual_images = set(f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    
    with open(ann_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    json_images_list = data.get('images', [])
    json_annotations = data.get('annotations', [])
    
    json_image_names = set(img['file_name'] for img in json_images_list)
    
    print(f"\nCOUNTS:")
    print(f" - Physical image files: {len(actual_images)}")
    print(f" - JSON image entries: {len(json_images_list)}")
    print(f" - JSON annotation entries: {len(json_annotations)}")
    
    missing_in_json = actual_images - json_image_names
    missing_on_disk = json_image_names - actual_images
    
    print(f"\nDISCREPANCIES:")
    print(f" - Images in JSON but missing on disk: {len(missing_on_disk)}")
    print(f" - Images on disk but missing from JSON: {len(missing_in_json)}")
    
    missing_on_disk_all.extend([f"[{split}] {f}" for f in missing_on_disk])
    missing_in_json_all.extend([f"[{split}] {f}" for f in missing_in_json])
    
    print(f"\nSAMPLING:")
    
    sample_json = random.sample(list(json_image_names), min(10, len(json_image_names)))
    print(" 10 Random JSON images and their on-disk status:")
    for f in sample_json:
        print(f"   - {f}: {'EXISTS' if f in actual_images else 'MISSING'}")
        
    sample_disk = random.sample(list(actual_images), min(10, len(actual_images)))
    print("\n 10 Random physical images and their JSON status:")
    for f in sample_disk:
        print(f"   - {f}: {'IN JSON' if f in json_image_names else 'MISSING FROM JSON'}")
    
    print("\n")

out_missing_on_disk = os.path.abspath("missing_on_disk.txt")
out_missing_in_json = os.path.abspath("missing_in_json.txt")

with open(out_missing_on_disk, "w", encoding="utf-8") as f:
    for item in missing_on_disk_all:
        f.write(item + "\n")

with open(out_missing_in_json, "w", encoding="utf-8") as f:
    for item in missing_in_json_all:
        f.write(item + "\n")

print(f"Saved missing files lists to:")
print(f" - {out_missing_on_disk}")
print(f" - {out_missing_in_json}")
