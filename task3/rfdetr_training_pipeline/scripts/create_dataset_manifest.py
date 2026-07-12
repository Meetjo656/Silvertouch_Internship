import os
import json
import yaml
import hashlib
from datetime import datetime
from pathlib import Path

def calculate_sha256(filepath):
    if not os.path.exists(filepath):
        return None
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def generate_manifest(workspace, config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    manifest = {
        "dataset_version": "v1.0",
        "creation_date": datetime.now().isoformat(),
        "number_of_images": 0,
        "number_of_annotations": 0,
        "number_of_classes": config.get("num_classes", 8),
        "class_distribution": {},
        "annotation_source": "roboflow/coco_export",
        "splits": {}
    }
    
    ontology = config.get("class_names", [])
    total_class_dist = {name: 0 for name in ontology}
    
    for split in ["train", "val", "test"]:
        ann_path = config.get(f"{split}_annotations")
        img_dir = config.get(f"{split}_images")
        
        if not ann_path or not os.path.exists(ann_path):
            continue
            
        with open(ann_path, 'r') as f:
            data = json.load(f)
            
        categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
        
        num_images = len(data.get("images", []))
        num_anns = len(data.get("annotations", []))
        
        manifest["number_of_images"] += num_images
        manifest["number_of_annotations"] += num_anns
        
        split_dist = {}
        for ann in data.get("annotations", []):
            cat_name = categories.get(ann["category_id"], "Unknown")
            split_dist[cat_name] = split_dist.get(cat_name, 0) + 1
            if cat_name in total_class_dist:
                total_class_dist[cat_name] += 1
                
        # Hash image list
        image_filenames = sorted([img["file_name"] for img in data.get("images", [])])
        image_list_str = "\n".join(image_filenames).encode('utf-8')
        img_hash = hashlib.sha256(image_list_str).hexdigest()
        
        manifest["splits"][split] = {
            "images": num_images,
            "annotations": num_anns,
            "class_distribution": split_dist,
            "annotation_sha256": calculate_sha256(ann_path),
            "image_list_sha256": img_hash
        }
        
    manifest["class_distribution"] = total_class_dist
    
    out_path = workspace / "dataset_manifest.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(manifest, f, sort_keys=False)
        
    print(f"Dataset manifest created at {out_path}")

def main():
    script_dir = Path(__file__).parent
    workspace = script_dir.parent
    config_path = workspace / "configs" / "dataset_config.yaml"
    generate_manifest(workspace, config_path)

if __name__ == "__main__":
    main()
