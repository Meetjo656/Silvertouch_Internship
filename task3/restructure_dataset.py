import os
import json
import shutil

def merge_jsons(json_paths, output_path):
    merged_data = {
        "licenses": [],
        "info": {},
        "categories": [],
        "images": [],
        "annotations": []
    }
    
    img_id_offset = 0
    ann_id_offset = 0
    
    first = True
    
    for jp in json_paths:
        if not os.path.exists(jp):
            continue
            
        with open(jp, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if first:
            merged_data["licenses"] = data.get("licenses", [])
            merged_data["info"] = data.get("info", {})
            merged_data["categories"] = data.get("categories", [])
            first = False
            
        # Map old image IDs to new ones
        img_id_map = {}
        max_img_id = 0
        for img in data.get("images", []):
            old_id = img["id"]
            new_id = old_id + img_id_offset
            img_id_map[old_id] = new_id
            
            new_img = img.copy()
            new_img["id"] = new_id
            merged_data["images"].append(new_img)
            
            if new_id > max_img_id:
                max_img_id = new_id
                
        max_ann_id = 0
        for ann in data.get("annotations", []):
            old_ann_id = ann["id"]
            new_ann_id = old_ann_id + ann_id_offset
            
            new_ann = ann.copy()
            new_ann["id"] = new_ann_id
            new_ann["image_id"] = img_id_map[ann["image_id"]]
            merged_data["annotations"].append(new_ann)
            
            if new_ann_id > max_ann_id:
                max_ann_id = new_ann_id
                
        img_id_offset = max_img_id
        ann_id_offset = max_ann_id
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f)
        
def restructure():
    base_dir = r"d:\Silvertouch Internship\task3\final_training_v2"
    
    # 1. Setup target dirs
    train_dir = os.path.join(base_dir, "train")
    valid_dir = os.path.join(base_dir, "valid")
    test_dir = os.path.join(base_dir, "test")
    
    os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "annotations"), exist_ok=True)
    
    # 2. Merge Train sets (New_Training, train (1), hemang_sir)
    train_sources = ["New_Training", "train (1)", "hemang_sir"]
    json_paths_to_merge = []
    
    for src in train_sources:
        src_dir = os.path.join(base_dir, src)
        if not os.path.exists(src_dir):
            continue
            
        # Copy images
        src_img_dir = os.path.join(src_dir, "images")
        if os.path.exists(src_img_dir):
            for f in os.listdir(src_img_dir):
                shutil.copy2(os.path.join(src_img_dir, f), os.path.join(train_dir, "images", f))
                
        # Find JSON
        src_ann_dir = os.path.join(src_dir, "annotations")
        if os.path.exists(src_ann_dir):
            jsons = [f for f in os.listdir(src_ann_dir) if f.endswith('.json')]
            if jsons:
                json_paths_to_merge.append(os.path.join(src_ann_dir, jsons[0]))
                
    if json_paths_to_merge:
        merge_jsons(json_paths_to_merge, os.path.join(train_dir, "annotations.json"))
        
    # 3. Rename Validation (1) to valid
    val_src = os.path.join(base_dir, "Validation (1)")
    if os.path.exists(val_src):
        os.rename(val_src, valid_dir)
        # Rename JSON
        val_ann_dir = os.path.join(valid_dir, "annotations")
        if os.path.exists(val_ann_dir):
            jsons = [f for f in os.listdir(val_ann_dir) if f.endswith('.json')]
            if jsons:
                os.rename(os.path.join(val_ann_dir, jsons[0]), os.path.join(valid_dir, "annotations.json"))
            # Remove old annotations dir if we moved the file up (or keep it in annotations.json, let's put it in root of valid/)
            # Wait, user said valid/annotations.json. Let's do that.
            shutil.rmtree(val_ann_dir, ignore_errors=True)
            
    # 4. Rename test (1) to test
    test_src = os.path.join(base_dir, "test (1)")
    if os.path.exists(test_src):
        os.rename(test_src, test_dir)
        # Rename JSON
        test_ann_dir = os.path.join(test_dir, "annotations")
        if os.path.exists(test_ann_dir):
            jsons = [f for f in os.listdir(test_ann_dir) if f.endswith('.json')]
            if jsons:
                os.rename(os.path.join(test_ann_dir, jsons[0]), os.path.join(test_dir, "annotations.json"))
            shutil.rmtree(test_ann_dir, ignore_errors=True)
            
    # Cleanup old train dirs
    for src in train_sources:
        src_dir = os.path.join(base_dir, src)
        if os.path.exists(src_dir):
            shutil.rmtree(src_dir, ignore_errors=True)
            
    print("Restructured final_training_v2 successfully.")

if __name__ == "__main__":
    restructure()
