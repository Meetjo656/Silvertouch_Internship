import json
import yaml
from pathlib import Path

def create_yaml(config_path="config/yolo_dataset.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    train_coco_json = paths["train_coco_json"]
    output_dir = Path(paths["output_dir"])

    with open(paths["train_coco_json"], "r", encoding="utf-8") as f:
        train_coco = json.load(f)
    
    valid_categories = [c for c in train_coco["categories"] if c["name"] not in ["Artwork", "not_artwork"]]
    
    category_map = {
        idx: category["name"]
        for idx, category in enumerate(valid_categories)
    }
    
    yolo_yaml_content = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(valid_categories),
        "names": category_map
    }

    output_yaml_path = output_dir / "data.yaml"
    with open(output_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yolo_yaml_content, f, sort_keys=False)

    print(f"Generated data.yaml at {output_yaml_path}")

if __name__ == "__main__":
    create_yaml()
