import os
import cv2
import json
import random
import shutil
import numpy as np
from pathlib import Path

def pad_and_shift(img, bboxes, pad_size=15):
    """Adds white padding and shifts bounding boxes."""
    h, w = img.shape[:2]
    # Add white padding
    padded_img = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
    padded_bboxes = []
    for bbox in bboxes:
        x, y, bw, bh = bbox
        padded_bboxes.append([x + pad_size, y + pad_size, bw, bh])
        
    return padded_img, padded_bboxes

def augment_image(img, bboxes):
    """
    Returns a dict of augmented images and their bboxes:
    - orig: The original (padded) image
    - gray: Grayscale version
    - rot180: 180 degree rotation
    - rot90cw: 90 degree clockwise
    - rot90ccw: 90 degree counter-clockwise
    """
    h, w = img.shape[:2]
    augs = {}
    
    # 1. Original
    augs['orig'] = (img.copy(), [list(b) for b in bboxes])
    
    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) # Convert back to 3 channels for consistency
    augs['gray'] = (gray, [list(b) for b in bboxes])
    
    # 3. Rotate 180
    rot180 = cv2.rotate(img, cv2.ROTATE_180)
    bboxes180 = []
    for (x, y, bw, bh) in bboxes:
        bboxes180.append([w - x - bw, h - y - bh, bw, bh])
    augs['rot180'] = (rot180, bboxes180)
    
    # 4. Rotate 90 CW
    rot90cw = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    bboxes90cw = []
    for (x, y, bw, bh) in bboxes:
        bboxes90cw.append([h - y - bh, x, bh, bw])
    augs['rot90cw'] = (rot90cw, bboxes90cw)
    
    # 5. Rotate 90 CCW
    rot90ccw = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    bboxes90ccw = []
    for (x, y, bw, bh) in bboxes:
        bboxes90ccw.append([y, w - x - bw, bh, bw])
    augs['rot90ccw'] = (rot90ccw, bboxes90ccw)
    
    return augs

def main():
    base_dir = Path(r"d:\Silvertouch Internship\task3")
    input_dir = base_dir / "final_training" / "final_1"
    
    anno_path = input_dir / "annotations" / "instances_Train.json"
    images_dir = input_dir / "images" / "Train"
    
    output_dir = base_dir / "final_training" / "processed_dataset"
    out_images_dir = output_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading original annotations...")
    with open(anno_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
        
    img_by_id = {img['id']: img for img in coco['images']}
    anns_by_img = {img['id']: [] for img in coco['images']}
    
    for ann in coco['annotations']:
        anns_by_img[ann['image_id']].append(ann)
        
    new_images = []
    new_annotations = []
    ann_id_counter = 1
    
    print("Processing and augmenting images...")
    processed_count = 0
    for img_id, img_info in img_by_id.items():
        img_name = img_info['file_name']
        img_path = images_dir / img_name
        
        if not img_path.exists():
            print(f"Warning: {img_path} not found.")
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Failed to read {img_path}.")
            continue
            
        bboxes = [ann['bbox'] for ann in anns_by_img[img_id]]
        cat_ids = [ann['category_id'] for ann in anns_by_img[img_id]]
        
        # 1. Pad and shift
        padded_img, padded_bboxes = pad_and_shift(img, bboxes, pad_size=15)
        
        # 2. Augment
        augmented_data = augment_image(padded_img, padded_bboxes)
        
        # 3. Save augmented versions and record in new COCO dict
        base_name = Path(img_name).stem
        ext = Path(img_name).suffix
        
        for aug_type, (aug_img, aug_bboxes) in augmented_data.items():
            out_name = f"{base_name}_{aug_type}{ext}"
            out_path = out_images_dir / out_name
            cv2.imwrite(str(out_path), aug_img)
            
            aug_h, aug_w = aug_img.shape[:2]
            
            new_img_id = len(new_images) + 1
            new_images.append({
                "id": new_img_id,
                "file_name": out_name,
                "width": aug_w,
                "height": aug_h,
                "original_id": img_id
            })
            
            for bbox, cat_id in zip(aug_bboxes, cat_ids):
                new_annotations.append({
                    "id": ann_id_counter,
                    "image_id": new_img_id,
                    "category_id": cat_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0
                })
                ann_id_counter += 1
                
        processed_count += 1
        if processed_count % 50 == 0:
            print(f"Processed {processed_count} original images...")
            
    print(f"Total processed original images: {processed_count}")
    print(f"Total augmented images: {len(new_images)}")
    
    print("\nSplitting dataset into 70% Train, 20% Test, 10% Valid...")
    
    # We should split by original image ID to avoid data leak across splits
    orig_ids = list(img_by_id.keys())
    random.seed(42)
    random.shuffle(orig_ids)
    
    num_orig = len(orig_ids)
    train_split = int(0.7 * num_orig)
    test_split = int(0.9 * num_orig) # 70 + 20
    
    train_ids = set(orig_ids[:train_split])
    test_ids = set(orig_ids[train_split:test_split])
    valid_ids = set(orig_ids[test_split:])
    
    splits = {
        "train": {"images": [], "annotations": [], "dir": output_dir / "train"},
        "test": {"images": [], "annotations": [], "dir": output_dir / "test"},
        "valid": {"images": [], "annotations": [], "dir": output_dir / "valid"}
    }
    
    for s in splits.values():
        (s["dir"] / "images").mkdir(parents=True, exist_ok=True)
        
    for img_data in new_images:
        orig_id = img_data.pop("original_id") # Remove tracking key
        
        if orig_id in train_ids:
            target = "train"
        elif orig_id in test_ids:
            target = "test"
        else:
            target = "valid"
            
        splits[target]["images"].append(img_data)
        
        # Move image to split dir
        src = out_images_dir / img_data["file_name"]
        dst = splits[target]["dir"] / "images" / img_data["file_name"]
        shutil.move(str(src), str(dst))
        
    # Remove the generic out_images_dir since we moved all images
    shutil.rmtree(out_images_dir)
    
    # Distribute annotations
    for ann in new_annotations:
        img_id = ann["image_id"]
        # Find which split has this image_id
        for s in splits.values():
            if any(i["id"] == img_id for i in s["images"]):
                s["annotations"].append(ann)
                break
                
    # Save COCO JSONs
    for split_name, split_data in splits.items():
        coco_split = {
            "images": split_data["images"],
            "annotations": split_data["annotations"],
            "categories": coco["categories"]
        }
        json_path = split_data["dir"] / f"instances_{split_name.capitalize()}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco_split, f, indent=4)
            
        print(f"Saved {split_name} split: {len(split_data['images'])} images, {len(split_data['annotations'])} annotations.")

if __name__ == '__main__':
    main()
