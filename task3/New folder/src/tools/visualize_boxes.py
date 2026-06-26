import cv2
import os
from pathlib import Path
import random

def visualize_yolo_boxes(dataset_dir="yolo_detection_dataset", split="train", num_samples=5):
    base_dir = Path(dataset_dir)
    images_dir = base_dir / split / "images"
    labels_dir = base_dir / split / "labels"
    output_dir = Path("visualized_samples")
    
    output_dir.mkdir(exist_ok=True)
    
    # Load class names if data.yaml exists
    yaml_path = base_dir / "data.yaml"
    names = {}
    if yaml_path.exists():
        import yaml
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
            names = data.get("names", {})
            
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # Filter out empty negative samples for visualization
    valid_images = []
    for img in images:
        label_path = labels_dir / Path(img).with_suffix(".txt").name
        if label_path.exists() and os.path.getsize(label_path) > 0:
            valid_images.append(img)
            
    samples = random.sample(valid_images, min(num_samples, len(valid_images)))
    
    for img_name in samples:
        img_path = images_dir / img_name
        label_path = labels_dir / Path(img_name).with_suffix(".txt").name
        
        image = cv2.imread(str(img_path))
        if image is None: continue
        h, w = image.shape[:2]
        
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                
                # Unnormalize
                x_center, y_center = cx * w, cy * h
                box_w, box_h = bw * w, bh * h
                
                x_min = int(x_center - box_w / 2)
                y_min = int(y_center - box_h / 2)
                x_max = int(x_center + box_w / 2)
                y_max = int(y_center + box_h / 2)
                
                cls_name = names.get(cls_id, str(cls_id))
                
                # Draw box
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(image, cls_name, (x_min, max(0, y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
        out_path = output_dir / f"rendered_{img_name}"
        cv2.imwrite(str(out_path), image)
        print(f"Saved rendered image to {out_path}")

if __name__ == "__main__":
    visualize_yolo_boxes(num_samples=10)
