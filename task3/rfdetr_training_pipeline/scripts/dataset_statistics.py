import os
import json
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_stats(json_path, ontology_names):
    if not os.path.exists(json_path):
        return None
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    
    stats = {
        "total_images": len(data.get("images", [])),
        "total_annotations": len(data.get("annotations", [])),
        "instances_per_class": Counter(),
        "bbox_areas": [],
        "aspect_ratios": [],
        "image_sizes": Counter(),
        "empty_images": 0,
        "filenames": set(),
        "duplicate_filenames": 0
    }
    
    image_to_anns = defaultdict(list)
    for ann in data.get("annotations", []):
        image_to_anns[ann["image_id"]].append(ann)
        cat_name = categories.get(ann["category_id"], "Unknown")
        stats["instances_per_class"][cat_name] += 1
        
        bbox = ann.get("bbox", [])
        if len(bbox) == 4:
            w, h = bbox[2], bbox[3]
            stats["bbox_areas"].append(w * h)
            if h > 0:
                stats["aspect_ratios"].append(w / h)
                
    for img in data.get("images", []):
        fname = img.get("file_name")
        if fname in stats["filenames"]:
            stats["duplicate_filenames"] += 1
        stats["filenames"].add(fname)
        
        w, h = img.get("width", 0), img.get("height", 0)
        stats["image_sizes"][f"{w}x{h}"] += 1
        
        if len(image_to_anns[img["id"]]) == 0:
            stats["empty_images"] += 1
            
    return stats

def format_stats(split_name, stats, ontology_names, report_lines):
    if not stats:
        report_lines.append(f"## {split_name} Split")
        report_lines.append("❌ Not found.\n")
        return
        
    report_lines.append(f"## {split_name} Split")
    report_lines.append(f"- **Total Images:** {stats['total_images']}")
    report_lines.append(f"- **Total Annotations:** {stats['total_annotations']}")
    report_lines.append(f"- **Empty Images:** {stats['empty_images']}")
    report_lines.append(f"- **Duplicate Filenames:** {stats['duplicate_filenames']}")
    
    report_lines.append("\n### Image Sizes")
    for size, count in stats["image_sizes"].most_common():
        report_lines.append(f"- `{size}`: {count}")
        
    report_lines.append("\n### Class Imbalance (Instances per class)")
    for name in ontology_names:
        count = stats["instances_per_class"].get(name, 0)
        pct = (count / max(1, stats['total_annotations'])) * 100
        report_lines.append(f"- **{name}**: {count} ({pct:.1f}%)")
        
    if stats["bbox_areas"]:
        areas = stats["bbox_areas"]
        report_lines.append("\n### BBox Statistics")
        report_lines.append(f"- **Avg Size:** {np.mean(areas):.1f} px^2")
        report_lines.append(f"- **Min Size:** {np.min(areas):.1f} px^2")
        report_lines.append(f"- **Max Size:** {np.max(areas):.1f} px^2")
        
    report_lines.append("\n---\n")

def main():
    script_dir = Path(__file__).parent
    workspace = script_dir.parent
    config_path = workspace / "configs" / "dataset_config.yaml"
    report_path = workspace / "documentation" / "dataset_statistics.md"
    
    config = load_config(config_path)
    ontology = config["class_names"]
    
    report_lines = [
        "# Dataset Statistics Report",
        ""
    ]
    
    splits = [
        ("Train", config.get("train_annotations")),
        ("Validation", config.get("val_annotations")),
        ("Test", config.get("test_annotations"))
    ]
    
    for name, ann_path in splits:
        if ann_path:
            stats = generate_stats(ann_path, ontology)
            format_stats(name, stats, ontology, report_lines)
            
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"Dataset statistics generated at {report_path}")

if __name__ == "__main__":
    main()
