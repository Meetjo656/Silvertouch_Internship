import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def get_majority_class(label_path):
    # Determine if the image is mostly artwork (0) or not_artwork (1)
    # If tie or empty, default to 1 (not_artwork)
    if not os.path.exists(label_path):
        return 1
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    counts = {0: 0, 1: 0}
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        try:
            cls_id = int(parts[0])
            if cls_id in counts:
                counts[cls_id] += 1
        except ValueError:
            pass
            
    if counts[0] > counts[1]:
        return 0
    else:
        return 1

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images.sort() # Ensure deterministic order
    
    # Stratify based on majority class in each image
    labels_for_stratification = []
    for img in images:
        lbl_name = os.path.splitext(img)[0] + '.txt'
        lbl_path = os.path.join(labels_dir, lbl_name)
        labels_for_stratification.append(get_majority_class(lbl_path))
        
    print(f"Total valid images to split: {len(images)}")
    
    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_imgs, val_test_imgs, train_lbls, val_test_lbls = train_test_split(
        images, labels_for_stratification, 
        test_size=val_test_ratio, 
        random_state=seed, 
        stratify=labels_for_stratification
    )
    
    # Second split: val vs test
    test_relative_ratio = test_ratio / val_test_ratio
    val_imgs, test_imgs = train_test_split(
        val_test_imgs, 
        test_size=test_relative_ratio, 
        random_state=seed, 
        stratify=val_test_lbls
    )
    
    splits = {
        'train': train_imgs,
        'val': val_imgs,
        'test': test_imgs
    }
    
    for split, split_images in splits.items():
        print(f"Creating {split} split with {len(split_images)} images...")
        split_img_dir = os.path.join(output_dir, split, 'images')
        split_lbl_dir = os.path.join(output_dir, split, 'labels')
        
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_lbl_dir, exist_ok=True)
        
        for img_name in tqdm(split_images):
            # Copy image
            src_img = os.path.join(images_dir, img_name)
            dst_img = os.path.join(split_img_dir, img_name)
            shutil.copy2(src_img, dst_img)
            
            # Copy label
            lbl_name = os.path.splitext(img_name)[0] + '.txt'
            src_lbl = os.path.join(labels_dir, lbl_name)
            if os.path.exists(src_lbl):
                dst_lbl = os.path.join(split_lbl_dir, lbl_name)
                shutil.copy2(src_lbl, dst_lbl)

if __name__ == '__main__':
    pipeline_dir = Path(__file__).resolve().parent.parent
    
    temp_images = pipeline_dir / "dataset" / "temp_images"
    temp_labels = pipeline_dir / "dataset" / "temp_labels"
    
    output_dir = pipeline_dir / "dataset"
    
    print("--- 2. Check that the split is stratified ---")
    print("Splitting dataset...")
    split_dataset(temp_images, temp_labels, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    
    print("Cleaning up temporary directories...")
    if temp_images.exists():
        shutil.rmtree(temp_images)
    if temp_labels.exists():
        shutil.rmtree(temp_labels)
    
    print("Dataset splitting complete.")
