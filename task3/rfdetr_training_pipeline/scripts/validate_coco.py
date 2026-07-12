import os
import json
from pathlib import Path

def validate_coco(dataset_dir, split):
    print(f"Validating {split} dataset...")
    
    img_dir = os.path.join(dataset_dir, split)
    json_path = os.path.join(dataset_dir, split, "_annotations.coco.json")
    
    if not os.path.exists(json_path):
        print(f"  [ERROR] JSON missing: {json_path}")
        return False
        
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [ERROR] Invalid JSON: {e}")
        return False
        
    required_keys = ["info", "categories", "images", "annotations"]
    for k in required_keys:
        if k not in data:
            print(f"  [ERROR] Missing required key '{k}' in JSON.")
            return False
            
    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]
    
    # Check valid category IDs
    cat_ids = set(c["id"] for c in categories)
    if len(cat_ids) != len(categories):
        print("  [ERROR] Duplicate category IDs found.")
        return False
        
    # Check unique image IDs and file existence
    img_ids = set()
    img_id_to_dims = {}
    
    for img in images:
        img_id = img.get("id")
        if img_id in img_ids:
            print(f"  [ERROR] Duplicate image ID found: {img_id}")
            return False
        img_ids.add(img_id)
        
        w, h = img.get("width"), img.get("height")
        if not w or not h or w <= 0 or h <= 0:
            print(f"  [ERROR] Invalid dimensions for image {img_id}: {w}x{h}")
            return False
            
        img_id_to_dims[img_id] = (w, h)
        
        file_name = img.get("file_name")
        full_path = os.path.join(img_dir, file_name)
        if not os.path.exists(full_path):
            print(f"  [ERROR] Physical image missing: {full_path}")
            return False
            
    # Check unique annotation IDs, bbox bounds, matching image IDs
    ann_ids = set()
    for ann in annotations:
        ann_id = ann.get("id")
        if ann_id in ann_ids:
            print(f"  [ERROR] Duplicate annotation ID found: {ann_id}")
            return False
        ann_ids.add(ann_id)
        
        img_id = ann.get("image_id")
        if img_id not in img_ids:
            print(f"  [ERROR] Annotation {ann_id} references missing image_id {img_id}")
            return False
            
        cat_id = ann.get("category_id")
        if cat_id not in cat_ids:
            print(f"  [ERROR] Annotation {ann_id} references invalid category_id {cat_id}")
            return False
            
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            print(f"  [ERROR] Invalid bbox format for annotation {ann_id}: {bbox}")
            return False
            
        x_min, y_min, bw, bh = bbox
        if bw <= 0 or bh <= 0:
            print(f"  [ERROR] Invalid bbox size for annotation {ann_id}: {bw}x{bh}")
            return False
            
        # Float arithmetic precision handling
        img_w, img_h = img_id_to_dims[img_id]
        if x_min < -0.1 or y_min < -0.1 or (x_min + bw) > (img_w + 0.1) or (y_min + bh) > (img_h + 0.1):
            print(f"  [ERROR] Bbox out of bounds for annotation {ann_id}: {bbox} on image {img_w}x{img_h}")
            return False
            
    print(f"  [SUCCESS] {split} dataset passed all COCO schema validations.")
    return True
    
def main():
    dataset_dir = str(Path(__file__).resolve().parents[1] / "dataset")
    all_passed = True
    
    for split in ["train", "valid", "test"]:
        passed = validate_coco(dataset_dir, split)
        if not passed:
            all_passed = False
            
    if all_passed:
        print("ALL DATASETS VALIDATED SUCCESSFULLY.")
        exit(0)
    else:
        print("VALIDATION FAILED.")
        exit(1)
        
if __name__ == "__main__":
    main()
