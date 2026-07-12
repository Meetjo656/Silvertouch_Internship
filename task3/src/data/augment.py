import os
import cv2
import yaml
import albumentations as A
from pathlib import Path

def get_augment_pipeline(config):
    aug_config = config["augmentation"]
    
    transforms = []
    
    # Safe Photometric Transforms - always enable
    transforms.extend([
        A.Blur(blur_limit=3, p=0.4),
        A.MedianBlur(blur_limit=3, p=0.4),
        A.RGBShift(p=0.4),
        A.HueSaturationValue(p=0.4),
        A.RandomBrightnessContrast(p=0.4),
        A.CLAHE(p=0.4),
        A.RandomGamma(p=0.4)
    ])
    
    if hasattr(A, 'GaussNoise'):
        transforms.append(A.GaussNoise(p=0.4))
    elif hasattr(A, 'GaussianNoise'):
        transforms.append(A.GaussianNoise(p=0.4))
        
    # Optional Geometric Transforms (Rotation)
    if aug_config.get("enable_rotations", False):
        angles = aug_config.get("rotation_angles", [])
        if 90 in angles or 180 in angles or 270 in angles:
            transforms.append(A.RandomRotate90(p=0.8)) # RandomRotate90 handles 90, 180, 270

    return A.Compose(transforms, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=0.0, min_visibility=0.0))

def augment_dataset(config_path="config/yolo_dataset.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # We enforce train only
    num_augmentations = config["augmentation"].get("num_augmentations", 3)
    output_dir = Path(config["paths"]["output_dir"])
    train_images_dir = output_dir / "train" / "images"
    train_labels_dir = output_dir / "train" / "labels"
    
    if not train_images_dir.exists():
        print("Train split not found in output_dir. Check conversion step.")
        return

    transform = get_augment_pipeline(config)

    original_images = [f for f in os.listdir(train_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    original_images = [f for f in original_images if "_aug" not in f]

    total_augmented = 0

    for img_name in original_images:
        img_path = train_images_dir / img_name
        label_name = Path(img_name).with_suffix(".txt").name
        label_path = train_labels_dir / label_name

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        bboxes = []
        class_ids = []

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                coords = [float(p) for p in parts[1:]]
                if len(coords) == 4:
                    bboxes.append(coords) # [cx, cy, w, h]
                    class_ids.append(class_id)

        for i in range(num_augmentations):
            try:
                transformed = transform(image=image, bboxes=bboxes, class_labels=class_ids)
                new_img = transformed['image']
                new_bboxes = transformed['bboxes']
                new_class_ids = transformed['class_labels']
            except Exception as e:
                print(f"Error augmenting {img_name}: {e}")
                continue

            base_name = Path(img_name).stem
            ext = Path(img_name).suffix
            aug_img_name = f"{base_name}_aug{i}{ext}"
            aug_label_name = f"{base_name}_aug{i}.txt"
            
            cv2.imwrite(str(train_images_dir / aug_img_name), new_img)

            labels_content = []
            for cls_id, bbox in zip(new_class_ids, new_bboxes):
                cx, cy, w, h = bbox
                # Constrain bounds explicitly just in case
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w = max(0.000001, min(1.0, w))
                h = max(0.000001, min(1.0, h))
                labels_content.append(f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            
            with open(train_labels_dir / aug_label_name, "w", encoding="utf-8") as f:
                f.write("\n".join(labels_content))
                
            total_augmented += 1

    print(f"Augmentation completed: Generated {total_augmented} augmented images in train split.")

if __name__ == "__main__":
    augment_dataset()
