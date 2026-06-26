import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import uuid

def get_majority_class(label_path):
    if not os.path.exists(label_path): return 0
    with open(label_path, 'r') as f: lines = f.readlines()
    counts = {}
    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        try:
            cid = int(parts[0])
            counts[cid] = counts.get(cid, 0) + 1
        except: pass
    if not counts: return 0
    return max(counts, key=counts.get)

def copy_existing_dataset(src_dir, dst_dir):
    print("Copying existing base dataset (ensuring 1:1 image-label mapping)...")
    for split in ['train', 'valid', 'test']:
        src_split = os.path.join(src_dir, split)
        dst_split = os.path.join(dst_dir, split)
        
        src_img_dir = os.path.join(src_split, 'images')
        src_lbl_dir = os.path.join(src_split, 'labels')
        dst_img_dir = os.path.join(dst_split, 'images')
        dst_lbl_dir = os.path.join(dst_split, 'labels')
        
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)
        
        if os.path.exists(src_img_dir) and os.path.exists(src_lbl_dir):
            for img_name in os.listdir(src_img_dir):
                base_name = os.path.splitext(img_name)[0]
                lbl_name = base_name + '.txt'
                
                src_img_path = os.path.join(src_img_dir, img_name)
                src_lbl_path = os.path.join(src_lbl_dir, lbl_name)
                
                # Ensure 1:1 mapping (drop orphans)
                if os.path.exists(src_lbl_path):
                    shutil.copy2(src_img_path, os.path.join(dst_img_dir, img_name))
                    shutil.copy2(src_lbl_path, os.path.join(dst_lbl_dir, lbl_name))

def split_and_merge_new_dataset(temp_images_dir, temp_labels_dir, dst_dir, seed=42):
    if not os.path.exists(temp_images_dir):
        return 0
        
    images = [f for f in os.listdir(temp_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()
    
    labels_for_stratification = []
    for img in images:
        lbl_name = os.path.splitext(img)[0] + '.txt'
        lbl_path = os.path.join(temp_labels_dir, lbl_name)
        labels_for_stratification.append(get_majority_class(lbl_path))
        
    print(f"Total new valid images to split: {len(images)}")
    
    val_test_ratio = 0.2
    try:
        train_imgs, val_test_imgs, train_lbls, val_test_lbls = train_test_split(
            images, labels_for_stratification, test_size=val_test_ratio, random_state=seed, stratify=labels_for_stratification
        )
        val_imgs, test_imgs = train_test_split(
            val_test_imgs, test_size=0.5, random_state=seed, stratify=val_test_lbls
        )
    except ValueError:
        train_imgs, val_test_imgs = train_test_split(images, test_size=val_test_ratio, random_state=seed)
        val_imgs, test_imgs = train_test_split(val_test_imgs, test_size=0.5, random_state=seed)
    
    splits = {'train': train_imgs, 'valid': val_imgs, 'test': test_imgs}
    
    duplicates_renamed = 0
    for split, split_images in splits.items():
        print(f"Merging {split} split with {len(split_images)} images...")
        dst_img_dir = os.path.join(dst_dir, split, 'images')
        dst_lbl_dir = os.path.join(dst_dir, split, 'labels')
        
        for img_name in tqdm(split_images):
            base_name, ext = os.path.splitext(img_name)
            lbl_name = base_name + '.txt'
            
            src_img = os.path.join(temp_images_dir, img_name)
            src_lbl = os.path.join(temp_labels_dir, lbl_name)
            
            dst_img_name = img_name
            dst_lbl_name = lbl_name
            
            if os.path.exists(os.path.join(dst_img_dir, dst_img_name)) or os.path.exists(os.path.join(dst_lbl_dir, dst_lbl_name)):
                suffix = f"_new_{uuid.uuid4().hex[:6]}"
                dst_img_name = base_name + suffix + ext
                dst_lbl_name = base_name + suffix + '.txt'
                duplicates_renamed += 1
                
            shutil.copy2(src_img, os.path.join(dst_img_dir, dst_img_name))
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, os.path.join(dst_lbl_dir, dst_lbl_name))
                
    return duplicates_renamed

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent.parent.parent
    existing_dataset = base_dir / "New folder" / "yolo_detection_dataset"
    pipeline_dir = Path(__file__).resolve().parent.parent
    final_dataset = pipeline_dir / "dataset"
    
    temp_new_images = pipeline_dir / "dataset" / "temp_new_dataset" / "images"
    temp_new_labels = pipeline_dir / "dataset" / "temp_new_dataset" / "labels"
    
    # Delete dataset to rebuild cleanly
    if os.path.exists(final_dataset):
        for split in ['train', 'valid', 'test']:
            split_dir = os.path.join(final_dataset, split)
            if os.path.exists(split_dir):
                shutil.rmtree(split_dir)
                
    copy_existing_dataset(existing_dataset, final_dataset)
    dups = split_and_merge_new_dataset(temp_new_images, temp_new_labels, final_dataset)
    
    print(f"Merge complete. Duplicates renamed: {dups}")
