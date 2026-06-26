import os
import json
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = r"d:\Silvertouch Internship\task3\trained_models\structural_8class_50epochs\weights\best.pt"
IMG_BASE_DIRS = {
    "Artwork": r"d:\Silvertouch Internship\task3\new (1)\images\Artwork",
    "Not_Artwork": r"d:\Silvertouch Internship\task3\new (1)\images\Not_Artwork"
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
OUTPUT_JSON = r"d:\Silvertouch Internship\task3\raw_confidences_all.json"

def main():
    print(f"Loading model {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    all_detections = []
    total_images = 0
    
    for subdir, img_dir in IMG_BASE_DIRS.items():
        print(f"Processing {subdir}...")
        files = [f for f in os.listdir(img_dir) if Path(f).suffix.lower() in IMAGE_EXTENSIONS and " - Copy" not in f]
        
        for i, f in enumerate(files):
            img_path = os.path.join(img_dir, f)
            results = model.predict(source=img_path, conf=0.01, iou=0.99, verbose=False, imgsz=640)
            
            total_images += 1
            boxes = results[0].boxes
            
            image_preds = []
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    image_preds.append({"class_id": cls_id, "conf": conf})
                    
            all_detections.append({
                "file_name": f,
                "subdir": subdir,
                "predictions": image_preds
            })
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(files)} in {subdir}")
                
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_detections, f, indent=2)
        
    print(f"Finished extracting from {total_images} images. Output saved to {OUTPUT_JSON}")

if __name__ == '__main__':
    main()
