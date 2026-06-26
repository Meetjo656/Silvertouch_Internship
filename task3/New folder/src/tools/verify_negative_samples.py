import os
import shutil
import random
from ultralytics import YOLO

def main():
    # Paths
    img_dir = r"D:\Silvertouch Internship\task3\Scrapped\output\images"
    output_dir = r"d:\Silvertouch Internship\task3\negative_sample_overlays"
    model_path = r"D:\Crop-Forge\runs\detect\runs\detect\structural_yolov8n_512_feasibility\weights\best.pt"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Select 5 images randomly
    all_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.seed(42) # fixed seed for reproducible reporting
    selected = random.sample(all_files, 5)
    print(f"SELECTED_FILES: {selected}")
    
    # Load model
    model = YOLO(model_path)
    
    # 2. Run inference
    for img in selected:
        src_path = os.path.join(img_dir, img)
        orig_dest = os.path.join(output_dir, img.replace('.jpg', '_original.jpg'))
        shutil.copy(src_path, orig_dest)
        
        # Inference (conf=0.1 to catch even low confidence detections)
        results = model.predict(source=src_path, imgsz=512, save=False, conf=0.1)
        res = results[0]
        
        # Save overlay
        overlay_dest = os.path.join(output_dir, img.replace('.jpg', '_prediction_overlay.jpg'))
        res.save(filename=overlay_dest)
        
        # Extract metrics
        num_dets = len(res.boxes)
        classes = [model.names[int(c)] for c in res.boxes.cls]
        confs = [float(c) for c in res.boxes.conf]
        max_conf = max(confs) if confs else 0.0
        
        print(f"IMAGE: {img} | DETECTIONS: {num_dets} | MAX_CONF: {max_conf:.3f} | CLASSES: {classes}")

if __name__ == '__main__':
    main()
