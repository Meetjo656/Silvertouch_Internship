import os
import shutil
from ultralytics import YOLO
import random

def main():
    # Model path from previous logs
    model_path = r"D:\Crop-Forge\runs\detect\runs\detect\structural_yolov8n_512_feasibility\weights\best.pt"
    
    # Dataset paths
    img_dir = r"d:\Silvertouch Internship\task3\yolo_detection_dataset\test\images"
    lbl_dir = r"d:\Silvertouch Internship\task3\yolo_detection_dataset\test\labels"
    output_dir = r"d:\Silvertouch Internship\task3\verification_overlays"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find diverse images
    # 0: Front_Panel, 1: Side_Panel, 2: Back_Panel, 3: Extra_Areas, 4: Bottom_Part, 5: Top_Panel
    candidates = []
    found_extra = False
    found_top = False
    found_bottom = False
    
    files = os.listdir(lbl_dir)
    random.seed(42) # deterministic for report writing
    random.shuffle(files)
    
    for f in files:
        if len(candidates) >= 5: break
        with open(os.path.join(lbl_dir, f)) as file:
            classes = [int(line.split()[0]) for line in file.read().strip().split('\n') if line.strip()]
        
        img_name = f.replace('.txt', '.jpg')
        if not os.path.exists(os.path.join(img_dir, img_name)):
            continue
            
        added = False
        if 3 in classes and not found_extra:
            candidates.append(img_name)
            found_extra = True
            added = True
        elif 5 in classes and not found_top:
            candidates.append(img_name)
            found_top = True
            added = True
        elif 4 in classes and not found_bottom:
            candidates.append(img_name)
            found_bottom = True
            added = True
        elif len(set(classes)) >= 3 and not added:
            if img_name not in candidates:
                candidates.append(img_name)
                
    print(f"Selected candidates: {candidates}")
    
    # Load model
    model = YOLO(model_path)
    
    for img in candidates:
        src_path = os.path.join(img_dir, img)
        orig_dest = os.path.join(output_dir, img.replace('.jpg', '_original.jpg'))
        shutil.copy(src_path, orig_dest)
        
        # Inference
        results = model.predict(source=src_path, imgsz=512, save=False)
        res = results[0]
        overlay_dest = os.path.join(output_dir, img.replace('.jpg', '_prediction_overlay.jpg'))
        res.save(filename=overlay_dest)
        
        # Print detected classes for report writing
        detected_classes = [model.names[int(c)] for c in res.boxes.cls]
        print(f"Processed {img}. Detected: {detected_classes}")
        
if __name__ == '__main__':
    main()
