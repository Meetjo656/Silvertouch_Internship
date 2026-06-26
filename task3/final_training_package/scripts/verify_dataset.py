import os
import sys
from pathlib import Path
from PIL import Image

def verify_dataset(pipeline_dir):
    dataset_dir = os.path.join(pipeline_dir, 'dataset')
    splits = ['train', 'valid', 'test']
    
    print("Starting dataset validation...")
    total_images = 0
    total_labels = 0
    missing_labels = 0
    corrupt_images = 0
    invalid_ids = set()
    
    for split in splits:
        img_dir = os.path.join(dataset_dir, split, 'images')
        lbl_dir = os.path.join(dataset_dir, split, 'labels')
        
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            continue
            
        images = set(os.listdir(img_dir))
        labels = set(os.listdir(lbl_dir))
        
        for img in images:
            total_images += 1
            base = os.path.splitext(img)[0]
            lbl = base + '.txt'
            
            if lbl not in labels:
                print(f"FAILED: Image {img} has no corresponding label.")
                sys.exit(1)
                
            lbl_path = os.path.join(lbl_dir, lbl)
            img_path = os.path.join(img_dir, img)
            
            try:
                with Image.open(img_path) as im:
                    im.verify()
            except Exception:
                print(f"FAILED: Corrupt image {img}")
                sys.exit(1)
                
            if os.path.getsize(lbl_path) == 0:
                print(f"FAILED: Empty label file {lbl}")
                sys.exit(1)
                
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"FAILED: Invalid YOLO format in {lbl}")
                        sys.exit(1)
                    cls_id = int(parts[0])
                    if cls_id < 0 or cls_id > 7:
                        print(f"FAILED: Invalid class ID {cls_id} in {lbl}")
                        invalid_ids.add(cls_id)
                        sys.exit(1)
                    
        for lbl in labels:
            total_labels += 1
            base = os.path.splitext(lbl)[0]
            if not any(base + ext in images for ext in ['.jpg', '.jpeg', '.png']):
                print(f"FAILED: Label {lbl} has no corresponding image.")
                sys.exit(1)

    print("Dataset validation PASSED!")
    
    doc_dir = os.path.join(pipeline_dir, 'documentation')
    os.makedirs(doc_dir, exist_ok=True)
    report = f"""# Dataset Validation Report

- Images: {total_images}
- Labels: {total_labels}
- Missing Labels: {missing_labels}
- Corrupt Images: {corrupt_images}
- Invalid Class IDs detected: {invalid_ids}

Validation: PASSED
"""
    with open(os.path.join(doc_dir, 'dataset_validation_report.md'), 'w') as f:
        f.write(report)

if __name__ == '__main__':
    pipeline_dir = Path(__file__).resolve().parent.parent
    verify_dataset(pipeline_dir)
