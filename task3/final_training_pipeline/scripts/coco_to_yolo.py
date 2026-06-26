import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def convert_coco_to_yolo(coco_json_path, images_dir, output_images_dir, output_labels_dir):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
        
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    print("--- 1. Verify the class mapping ---")
    print("Categories in COCO JSON:")
    for cat in coco_data.get('categories', []):
        print(f" {{'id': {cat['id']}, 'name': '{cat['name']}'}}")
    print("-----------------------------------\n")

    # Map strictly by NAME, not by ID
    category_mapping = {}
    for cat in coco_data.get('categories', []):
        if cat['name'] == 'Artwork':
            category_mapping[cat['id']] = 0
        elif cat['name'] == 'not_artwork':
            category_mapping[cat['id']] = 1

    images_info = {img['id']: img for img in coco_data.get('images', [])}
    
    # Process annotations
    annotations_by_image = {}
    for ann in coco_data.get('annotations', []):
        if ann['category_id'] not in category_mapping:
            continue
        
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
            
        annotations_by_image[img_id].append(ann)
        
    # Copy images and create YOLO label files
    print(f"Processing images...")
    valid_images = 0
    artwork_boxes = 0
    not_artwork_boxes = 0
    
    for img_id, img_info in tqdm(images_info.items()):
        # SKIP if there are no annotations for this image (avoid empty label files)
        if img_id not in annotations_by_image or len(annotations_by_image[img_id]) == 0:
            continue
            
        file_name = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        src_image_path = os.path.join(images_dir, file_name)
        if not os.path.exists(src_image_path):
            print(f"Warning: Image {src_image_path} not found.")
            continue
            
        # Copy image
        dst_image_path = os.path.join(output_images_dir, file_name)
        shutil.copy2(src_image_path, dst_image_path)
        
        # Write label file
        label_file_name = os.path.splitext(file_name)[0] + '.txt'
        label_file_path = os.path.join(output_labels_dir, label_file_name)
        
        valid_images += 1
        with open(label_file_path, 'w') as f:
            for ann in annotations_by_image[img_id]:
                # Convert to YOLO format
                cat_id = category_mapping[ann['category_id']]
                
                if cat_id == 0:
                    artwork_boxes += 1
                elif cat_id == 1:
                    not_artwork_boxes += 1
                    
                bbox = ann['bbox'] # [x_min, y_min, width, height]
                
                x_center = (bbox[0] + bbox[2] / 2.0) / width
                y_center = (bbox[1] + bbox[3] / 2.0) / height
                w = bbox[2] / width
                h = bbox[3] / height
                
                # Ensure coordinates are between 0 and 1
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))
                
                f.write(f"{cat_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print("\n--- 4. Verify class balance ---")
    print(f"Artwork boxes     : {artwork_boxes}")
    print(f"Not_artwork boxes : {not_artwork_boxes}")
    print(f"Valid images kept : {valid_images}")
    print("-------------------------------\n")

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent.parent
    coco_json = base_dir / "final_training" / "New_Training" / "annotations" / "instances_Train.json"
    images_dir = base_dir / "final_training" / "New_Training" / "images" / "Train"
    
    pipeline_dir = Path(__file__).resolve().parent.parent
    out_images = pipeline_dir / "dataset" / "temp_images"
    out_labels = pipeline_dir / "dataset" / "temp_labels"
    
    print("Starting COCO to YOLO conversion...")
    convert_coco_to_yolo(coco_json, images_dir, out_images, out_labels)
    print("Conversion complete.")
