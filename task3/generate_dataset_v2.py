import os
import json
import cv2
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

def process_dataset():
    input_base = r"d:\Silvertouch Internship\task3\final_training"
    output_base = r"d:\Silvertouch Internship\task3\final_training_v2"
    vis_base = r"d:\Silvertouch Internship\task3\dataset_visualization"
    
    pad_amount = 15
    
    # Create directories
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(os.path.join(vis_base, "boxes"), exist_ok=True)
    os.makedirs(os.path.join(vis_base, "boxes_and_names"), exist_ok=True)
    
    splits = [d for d in os.listdir(input_base) if os.path.isdir(os.path.join(input_base, d))]
    
    stats = {
        "total_images": 0,
        "images_per_split": {},
        "total_annotations": 0,
        "class_distribution": {},
        "bbox_sizes": [],
        "artwork_coverages": [],
        "only_not_artwork": 0,
        "multiple_panels": 0,
        "failed": 0,
        "processed": 0
    }
    
    for split in splits:
        print(f"Processing split: {split}")
        input_split_dir = os.path.join(input_base, split)
        out_split_dir = os.path.join(output_base, split)
        
        img_in_dir = os.path.join(input_split_dir, "images")
        ann_in_dir = os.path.join(input_split_dir, "annotations")
        
        if not os.path.exists(img_in_dir) or not os.path.exists(ann_in_dir):
            continue
            
        img_out_dir = os.path.join(out_split_dir, "images")
        ann_out_dir = os.path.join(out_split_dir, "annotations")
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(ann_out_dir, exist_ok=True)
        
        stats["images_per_split"][split] = 0
        
        # Find json
        json_files = [f for f in os.listdir(ann_in_dir) if f.endswith('.json')]
        if not json_files:
            continue
            
        ann_path = os.path.join(ann_in_dir, json_files[0])
        
        with open(ann_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Map categories
        cat_id_to_name = {c['id']: c['name'] for c in data['categories']}
        
        not_artwork_id = None
        artwork_id = None
        structural_ids = []
        
        for cat in data['categories']:
            name = cat['name'].lower()
            if name == "not_artwork":
                not_artwork_id = cat['id']
            elif name == "artwork":
                artwork_id = cat['id']
            else:
                structural_ids.append(cat['id'])
                
        # Process images
        img_dict = {img['id']: img for img in data['images']}
        ann_dict = {}
        for ann in data['annotations']:
            ann_dict.setdefault(ann['image_id'], []).append(ann)
            
        new_images = []
        new_annotations = []
        
        for img_info in tqdm(data['images'], desc=split):
            img_id = img_info['id']
            file_name = img_info['file_name']
            img_path = None
            for root, _, files in os.walk(img_in_dir):
                if file_name in files:
                    img_path = os.path.join(root, file_name)
                    break
            
            if img_path is None or not os.path.exists(img_path):
                stats["failed"] += 1
                continue
                
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                stats["failed"] += 1
                continue
                
            orig_h, orig_w = img.shape[:2]
            
            # Add padding
            padded_img = cv2.copyMakeBorder(
                img, pad_amount, pad_amount, pad_amount, pad_amount,
                cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
            
            new_h, new_w = padded_img.shape[:2]
            
            # Save padded image
            out_img_path = os.path.join(img_out_dir, file_name)
            cv2.imwrite(out_img_path, padded_img)
            
            # Update image info
            new_img_info = img_info.copy()
            new_img_info['width'] = new_w
            new_img_info['height'] = new_h
            new_images.append(new_img_info)
            
            anns = ann_dict.get(img_id, [])
            
            # Track classes for stats
            classes_in_img = set(a['category_id'] for a in anns)
            if len(classes_in_img) == 1 and not_artwork_id in classes_in_img:
                stats["only_not_artwork"] += 1
                
            panel_count = sum(1 for a in anns if a['category_id'] in structural_ids)
            if panel_count > 1:
                stats["multiple_panels"] += 1
                
            # Compute union for Artwork BEFORE shifting coordinates
            union_x_min = float('inf')
            union_y_min = float('inf')
            union_x_max = float('-inf')
            union_y_max = float('-inf')
            
            has_structural = False
            
            for ann in anns:
                if ann['category_id'] in structural_ids:
                    x, y, w, h = ann['bbox']
                    union_x_min = min(union_x_min, x)
                    union_y_min = min(union_y_min, y)
                    union_x_max = max(union_x_max, x + w)
                    union_y_max = max(union_y_max, y + h)
                    has_structural = True
            
            artwork_bbox = None
            if has_structural:
                aw = union_x_max - union_x_min
                ah = union_y_max - union_y_min
                artwork_bbox = [union_x_min, union_y_min, aw, ah]
            
            final_anns_for_img = []
            for ann in anns:
                new_ann = ann.copy()
                
                # Replace artwork bbox if recomputed
                if new_ann['category_id'] == artwork_id and artwork_bbox is not None:
                    new_ann['bbox'] = artwork_bbox
                    
                # Shift coordinates (we shift AFTER potentially updating the artwork box)
                x, y, w, h = new_ann['bbox']
                new_x = x + pad_amount
                new_y = y + pad_amount
                new_ann['bbox'] = [new_x, new_y, w, h]
                
                final_anns_for_img.append(new_ann)
                new_annotations.append(new_ann)
                
                stats["total_annotations"] += 1
                cat_name = cat_id_to_name.get(new_ann['category_id'], "Unknown")
                stats["class_distribution"][cat_name] = stats["class_distribution"].get(cat_name, 0) + 1
                stats["bbox_sizes"].append(w * h)
                
                if new_ann['category_id'] == artwork_id:
                    stats["artwork_coverages"].append((w * h) / (new_w * new_h))
            
            stats["total_images"] += 1
            stats["images_per_split"][split] += 1
            stats["processed"] += 1
            
            # Visualizations
            vis_img_boxes = padded_img.copy()
            vis_img_names = padded_img.copy()
            
            for ann in final_anns_for_img:
                if ann['category_id'] == not_artwork_id:
                    continue
                    
                x, y, w, h = map(int, map(round, ann['bbox']))
                cat_name = cat_id_to_name.get(ann['category_id'], "Unknown")
                
                # Colors based on class (deterministic pseudo-random)
                np.random.seed(ann['category_id'])
                color = tuple(int(c) for c in np.random.randint(0, 255, 3))
                
                cv2.rectangle(vis_img_boxes, (x, y), (x + w, y + h), color, 2)
                
                cv2.rectangle(vis_img_names, (x, y), (x + w, y + h), color, 2)
                # Text background
                (tw, th), _ = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis_img_names, (x, y - th - 5), (x + tw, y), color, -1)
                cv2.putText(vis_img_names, cat_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                
            vis_filename = f"{split}_{file_name}"
            cv2.imwrite(os.path.join(vis_base, "boxes", vis_filename), vis_img_boxes)
            cv2.imwrite(os.path.join(vis_base, "boxes_and_names", vis_filename), vis_img_names)
            
        # Save new JSON
        data['images'] = new_images
        data['annotations'] = new_annotations
        with open(os.path.join(ann_out_dir, json_files[0]), 'w', encoding='utf-8') as f:
            json.dump(data, f)
            
    # Generate Reports
    generate_reports(stats)
    
def generate_reports(stats):
    stats_path = r"d:\Silvertouch Internship\task3\dataset_statistics.md"
    pad_path = r"d:\Silvertouch Internship\task3\padding_report.md"
    
    if stats["bbox_sizes"]:
        avg_bbox = sum(stats["bbox_sizes"]) / len(stats["bbox_sizes"])
        max_bbox = max(stats["bbox_sizes"])
        min_bbox = min(stats["bbox_sizes"])
    else:
        avg_bbox = max_bbox = min_bbox = 0
        
    avg_art_cov = (sum(stats["artwork_coverages"]) / len(stats["artwork_coverages"])) * 100 if stats["artwork_coverages"] else 0
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("# Dataset Statistics\n\n")
        f.write(f"- **Total Images**: {stats['total_images']}\n")
        f.write("- **Images per Split**:\n")
        for s, c in stats["images_per_split"].items():
            f.write(f"  - {s}: {c}\n")
        f.write(f"- **Total Annotations**: {stats['total_annotations']}\n")
        f.write("- **Class Distribution**:\n")
        for c, count in stats["class_distribution"].items():
            f.write(f"  - {c}: {count}\n")
        f.write(f"- **Average BBox Area**: {avg_bbox:.2f} px^2\n")
        f.write(f"- **Largest BBox Area**: {max_bbox:.2f} px^2\n")
        f.write(f"- **Smallest BBox Area**: {min_bbox:.2f} px^2\n")
        f.write(f"- **Average Artwork Coverage**: {avg_art_cov:.2f}%\n")
        f.write(f"- **Images with only 'not_artwork'**: {stats['only_not_artwork']}\n")
        f.write(f"- **Images containing multiple panels**: {stats['multiple_panels']}\n")
        
    with open(pad_path, 'w', encoding='utf-8') as f:
        f.write("# Padding Report\n\n")
        f.write("- **Padding Added**: 15 px (all sides)\n")
        f.write("- **Coordinate Shift**: (+15, +15)\n")
        f.write(f"- **Images Processed Successfully**: {stats['processed']}\n")
        f.write(f"- **Images Failed (Not Found/Corrupt)**: {stats['failed']}\n")

if __name__ == "__main__":
    process_dataset()
