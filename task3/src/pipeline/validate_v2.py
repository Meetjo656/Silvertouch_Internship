from ultralytics import YOLO
import os
import json
from pathlib import Path

def main():
    base_dir = Path("d:/Silvertouch Internship/task3")
    # Path to the newly trained model
    model_path = base_dir / "trained_models/structural_8class_round2/weights/best.pt"
    
    # Validation dataset
    val_images_dir = base_dir / "dataset_v2/images"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
        
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    print("Running inference on dataset_v2...")
    
    front_panel_count = 0
    back_panel_count = 0
    ocr_ready_count = 0
    
    confidence_distribution = {
        "0.0-0.5": 0,
        "0.5-0.7": 0,
        "0.7-0.9": 0,
        "0.9-1.0": 0
    }
    
    # Using correct 8-class mapping
    # 0: Artwork, 1: Not_Artwork, 2: Front_Panel, 3: Side_Panel, 4: Back_Panel, 5: Extra_Areas, 6: Bottom_Part, 7: Top_Panel
    
    total_images = 0
    
    for img_file in val_images_dir.iterdir():
        if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
            
        total_images += 1
        
        # predict
        results = model.predict(str(img_file), verbose=False, imgsz=800)
        
        has_front = False
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # Update confidence distribution
                if conf < 0.5:
                    confidence_distribution["0.0-0.5"] += 1
                elif conf < 0.7:
                    confidence_distribution["0.5-0.7"] += 1
                elif conf < 0.9:
                    confidence_distribution["0.7-0.9"] += 1
                else:
                    confidence_distribution["0.9-1.0"] += 1
                    
                if cls_id == 2:  # Front_Panel
                    front_panel_count += 1
                    has_front = True
                elif cls_id == 4:  # Back_Panel
                    back_panel_count += 1
                    
        if has_front:
            ocr_ready_count += 1
            
    ocr_ready_rate = (ocr_ready_count / total_images) if total_images > 0 else 0
            
    report = {
        "total_images_processed": total_images,
        "Front_Panel_detections": front_panel_count,
        "Back_Panel_detections": back_panel_count,
        "ocr_ready_images": ocr_ready_count,
        "ocr_ready_rate": ocr_ready_rate,
        "confidence_distribution": confidence_distribution
    }
    
    report_path = base_dir / "structural_detection_report_v2.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Validation complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()
