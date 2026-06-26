import json
import os
import shutil
import yaml
import sys
from pathlib import Path
from tqdm import tqdm

def load_yolo_ontology(data_yaml_path):
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('names', {})

def convert_coco_to_yolo(coco_json_path, data_yaml_path, images_dir, output_images_dir, output_labels_dir):
    yolo_ontology = load_yolo_ontology(data_yaml_path)
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
        
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    print("--- Checking Mappings ---")
    coco_categories = coco_data.get('categories', [])
    
    # Normalization map for minor discrepancies
    normalization = {
        'not_artwork': 'Not_Artwork',
        'Bottom_Panel': 'Bottom_Part'
    }
    
    # Build a reverse lookup for YOLO ontology: name -> ID
    yolo_name_to_id = {v: k for k, v in yolo_ontology.items()}
    
    category_mapping = {}
    unknown_classes = []
    
    for cat in coco_categories:
        coco_name = cat['name']
        normalized_name = normalization.get(coco_name, coco_name)
        
        if normalized_name in yolo_name_to_id:
            yolo_id = yolo_name_to_id[normalized_name]
            
            # Check for conflict: Does the COCO ID map differently natively? 
            # In COCO, category IDs are arbitrary, so we map COCO ID -> YOLO ID based on name.
            category_mapping[cat['id']] = yolo_id
            print(f"Mapped COCO '{coco_name}' (ID {cat['id']}) -> YOLO '{normalized_name}' (ID {yolo_id})")
        else:
            unknown_classes.append(coco_name)
            
    if unknown_classes:
        print(f"WARNING: Found unknown classes in COCO dataset that are NOT in YOLO ontology: {unknown_classes}")
        print("These annotations will be ignored.")

    images_info = {img['id']: img for img in coco_data.get('images', [])}
    
    annotations_by_image = {}
    for ann in coco_data.get('annotations', []):
        if ann['category_id'] not in category_mapping:
            continue
        
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
            
        annotations_by_image[img_id].append(ann)
        
    print(f"\nProcessing {len(images_info)} images...")
    valid_images = 0
    
    for img_id, img_info in tqdm(images_info.items()):
        if img_id not in annotations_by_image or len(annotations_by_image[img_id]) == 0:
            continue
            
        file_name = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        src_image_path = os.path.join(images_dir, file_name)
        if not os.path.exists(src_image_path):
            print(f"Warning: Image {src_image_path} not found.")
            continue
            
        dst_image_path = os.path.join(output_images_dir, file_name)
        shutil.copy2(src_image_path, dst_image_path)
        
        label_file_name = os.path.splitext(file_name)[0] + '.txt'
        label_file_path = os.path.join(output_labels_dir, label_file_name)
        
        valid_images += 1
        with open(label_file_path, 'w') as f:
            for ann in annotations_by_image[img_id]:
                cat_id = category_mapping[ann['category_id']]
                bbox = ann['bbox'] # [x_min, y_min, width, height]
                
                x_center = (bbox[0] + bbox[2] / 2.0) / width
                y_center = (bbox[1] + bbox[3] / 2.0) / height
                w = bbox[2] / width
                h = bbox[3] / height
                
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))
                
                f.write(f"{cat_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent.parent
    coco_json = base_dir / "final_training" / "New_Training" / "annotations" / "instances_Train.json"
    images_dir = base_dir / "final_training" / "New_Training" / "images" / "Train"
    data_yaml = base_dir / "New folder" / "yolo_detection_dataset" / "data.yaml"
    
    pipeline_dir = Path(__file__).resolve().parent.parent
    out_images = pipeline_dir / "dataset" / "temp_new_dataset" / "images"
    out_labels = pipeline_dir / "dataset" / "temp_new_dataset" / "labels"
    
    print("Starting COCO to YOLO conversion (8-class preservation)...")
    convert_coco_to_yolo(coco_json, data_yaml, images_dir, out_images, out_labels)
    print("Conversion complete.")
