import os
import json
import shutil
import random
from ultralytics import YOLO

# Configurations
JSON_PATH = r"d:\Silvertouch Internship\task3\dataset_split\assembled_cartons\annotations_assembled.json"
IMG_DIR = r"d:\Silvertouch Internship\task3\dataset_split\assembled_cartons\images"
YOLO_DIR = r"d:\Silvertouch Internship\task3\dataset_split\yolo_format"
MODEL_A_PATH = r"d:\Silvertouch Internship\task3\trained_models\structural_8class_50epochs\weights\best.pt"

# Create YOLO directories
os.makedirs(os.path.join(YOLO_DIR, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(YOLO_DIR, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(YOLO_DIR, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(YOLO_DIR, "labels", "val"), exist_ok=True)

def convert_coco_to_yolo():
    print("Converting COCO JSON to YOLO format...")
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
        
    img_by_id = {img["id"]: img for img in data["images"]}
    anns_by_img = {img["id"]: [] for img in data["images"]}
    for ann in data["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)
        
    # Split 80/20
    img_ids = list(img_by_id.keys())
    random.seed(42)
    random.shuffle(img_ids)
    
    split_idx = int(len(img_ids) * 0.8)
    train_ids = set(img_ids[:split_idx])
    val_ids = set(img_ids[split_idx:])
    
    for img_id, img_info in img_by_id.items():
        split = "train" if img_id in train_ids else "val"
        filename = img_info["file_name"]
        
        # Copy image
        src_img = os.path.join(IMG_DIR, filename)
        dst_img = os.path.join(YOLO_DIR, "images", split, filename)
        if os.path.exists(src_img) and not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)
            
        # Write label
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        dst_txt = os.path.join(YOLO_DIR, "labels", split, txt_filename)
        
        width = img_info["width"]
        height = img_info["height"]
        
        with open(dst_txt, "w") as f:
            for ann in anns_by_img[img_id]:
                # COCO: [x_min, y_min, width, height]
                x_min, y_min, box_w, box_h = ann["bbox"]
                
                # YOLO: [class_id, x_center, y_center, width, height] normalized
                x_center = (x_min + box_w / 2) / width
                y_center = (y_min + box_h / 2) / height
                norm_w = box_w / width
                norm_h = box_h / height
                
                # YOLO requires 0-indexed classes. In COCO, ids are 1-8.
                yolo_cls_id = ann["category_id"] - 1 
                
                f.write(f"{yolo_cls_id} {x_center} {y_center} {norm_w} {norm_h}\n")
                
    # Create YAML
    yaml_path = os.path.join(YOLO_DIR, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {YOLO_DIR}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n\n")
        f.write("names:\n")
        # Ensure mapping is correct (1=Artwork, 8=Top_Panel)
        classes = sorted(data["categories"], key=lambda x: x["id"])
        for c in classes:
            f.write(f"  {c['id']-1}: {c['name']}\n")
            
    return yaml_path

def main():
    yaml_path = convert_coco_to_yolo()
    
    print("\n--- Training Model B (Assembled Only) ---")
    # Using YOLOv8n (nano model) for fast validation
    model_b = YOLO("yolov8n.pt") 
    
    # Train for 20 epochs
    results_b = model_b.train(data=yaml_path, epochs=20, imgsz=640, name="assembled_model_val")
    model_b_path = model_b.trainer.best
    print(f"Model B trained and saved to: {model_b_path}")
    
    print("\n--- Comparing Models on Validation Set ---")
    
    # Evaluate Model A
    print("\nEvaluating Model A (Original Dataset Model)...")
    model_a = YOLO(MODEL_A_PATH)
    metrics_a = model_a.val(data=yaml_path, imgsz=640)
    
    # Evaluate Model B
    print("\nEvaluating Model B (Assembled Only Model)...")
    model_b_eval = YOLO(model_b_path)
    metrics_b = model_b_eval.val(data=yaml_path, imgsz=640)
    
    # Extract map50
    map50_a = metrics_a.box.map50
    map50_b = metrics_b.box.map50
    
    print("\n=== FINAL COMPARISON ===")
    print(f"Model A (Original) mAP50 : {map50_a:.4f}")
    print(f"Model B (Assembled) mAP50: {map50_b:.4f}")
    
    if map50_b > map50_a:
        print("\n-> Model B outperformed Model A. The ontology-conflict hypothesis is proven!")
    else:
        print("\n-> Model B did not outperform Model A. Further investigation needed.")

if __name__ == '__main__':
    main()
