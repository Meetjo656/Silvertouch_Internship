import os
import cv2
import json
import random
import shutil
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def main():
    base_dir = r"d:\Silvertouch Internship\task3"
    input_dir = os.path.join(base_dir, "Only Images")
    model_path = os.path.join(base_dir, r"trained_models\structural_8class_20epochs\weights\20_epochs_8classes.pt")
    
    # Output directories
    outputs_dir = os.path.join(base_dir, "outputs")
    detections_dir = os.path.join(outputs_dir, "detections")
    stats_dir = os.path.join(outputs_dir, "statistics")
    
    overlays_dir = os.path.join(outputs_dir, "overlays")
    random_samples_dir = os.path.join(overlays_dir, "random_samples")
    no_fp_dir = os.path.join(overlays_dir, "no_front_panel")
    no_bp_dir = os.path.join(overlays_dir, "no_back_panel")
    low_conf_dir = os.path.join(overlays_dir, "low_confidence")
    
    for d in [detections_dir, stats_dir, random_samples_dir, no_fp_dir, no_bp_dir, low_conf_dir]:
        ensure_dir(d)
        
    print("Loading YOLO model...")
    model = YOLO(model_path)
    class_names = model.names
    
    all_images = []
    if os.path.exists(input_dir):
        for f in os.listdir(input_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(input_dir, f))
                
    print(f"Found {len(all_images)} images for structural audit.")
    
    # Statistics Tracking
    class_stats = {name: {"detections": 0, "images": 0, "confidences": []} for name in class_names.values()}
    
    artwork_stats = {
        "images_with_artwork": 0,
        "images_with_not_artwork": 0,
        "images_with_both": 0,
        "images_with_neither": 0
    }
    
    global_stats = {
        "images_processed": 0,
        "ocr_ready_images": 0,
        "images_with_front_panel": 0,
        "images_with_back_panel": 0,
        "images_missing_front_panel": 0,
        "images_missing_back_panel": 0
    }
    
    processed_overlays = []
    
    # Generate distinct colors for 8 classes
    np.random.seed(42)
    colors = {name: (random.randint(0,255), random.randint(0,255), random.randint(0,255)) for name in class_names.values()}
    
    for idx, img_path in enumerate(all_images):
        filename = os.path.basename(img_path)
        print(f"[{idx+1}/{len(all_images)}] Processing: {filename}")
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        annotated_img = img.copy()
        results = model(img, verbose=False)
        
        detections = []
        img_classes_detected = set()
        min_conf_in_image = 1.0
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = class_names[cls_id]
                conf = round(float(box.conf[0].item()), 4)
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                detections.append({
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })
                
                img_classes_detected.add(cls_name)
                class_stats[cls_name]["detections"] += 1
                class_stats[cls_name]["confidences"].append(conf)
                
                if conf < min_conf_in_image:
                    min_conf_in_image = conf
                
                # Draw box
                color = colors.get(cls_name, (0,255,0))
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(annotated_img, label, (x1, max(10, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        # Phase 3: Save Raw Detections
        det_json_path = os.path.join(detections_dir, f"{os.path.splitext(filename)[0]}.json")
        with open(det_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "image": filename,
                "detections": detections
            }, f, indent=4)
            
        # Update Class Image Counts
        for cls_name in img_classes_detected:
            class_stats[cls_name]["images"] += 1
            
        # Boolean Flags
        has_fp = "Front_Panel" in img_classes_detected
        has_bp = "Back_Panel" in img_classes_detected
        has_art = "Artwork" in img_classes_detected
        has_not_art = "Not_Artwork" in img_classes_detected
        
        # Global Stats
        global_stats["images_processed"] += 1
        
        if has_fp: global_stats["images_with_front_panel"] += 1
        else: global_stats["images_missing_front_panel"] += 1
            
        if has_bp: global_stats["images_with_back_panel"] += 1
        else: global_stats["images_missing_back_panel"] += 1
            
        if has_fp or has_bp:
            global_stats["ocr_ready_images"] += 1
            
        # Artwork Stats
        if has_art and has_not_art: artwork_stats["images_with_both"] += 1
        elif has_art: artwork_stats["images_with_artwork"] += 1
        elif has_not_art: artwork_stats["images_with_not_artwork"] += 1
        else: artwork_stats["images_with_neither"] += 1
            
        # Save Overlays based on rules (Phase 7)
        # Keep track of all generated overlay paths for the random selection
        overlay_temp_path = os.path.join(overlays_dir, f"temp_{filename}")
        cv2.imwrite(overlay_temp_path, annotated_img)
        processed_overlays.append(overlay_temp_path)
        
        if not has_fp:
            shutil.copy2(overlay_temp_path, os.path.join(no_fp_dir, filename))
        if not has_bp:
            shutil.copy2(overlay_temp_path, os.path.join(no_bp_dir, filename))
        if len(detections) > 0 and min_conf_in_image < 0.50:
            shutil.copy2(overlay_temp_path, os.path.join(low_conf_dir, filename))

    # Phase 8: Random Validation Set
    random.shuffle(processed_overlays)
    sampled_overlays = processed_overlays[:100]
    for overlay_path in sampled_overlays:
        shutil.copy2(overlay_path, os.path.join(random_samples_dir, os.path.basename(overlay_path).replace("temp_", "")))
        
    # Cleanup temp overlays
    for overlay_path in processed_overlays:
        if os.path.exists(overlay_path):
            os.remove(overlay_path)

    # Phase 4: Compile Final Class Statistics
    final_class_stats = {}
    for cls_name, data in class_stats.items():
        confs = data["confidences"]
        if len(confs) > 0:
            final_class_stats[cls_name] = {
                "detections": data["detections"],
                "images_containing_class": data["images"],
                "mean_confidence": round(float(np.mean(confs)), 4),
                "median_confidence": round(float(np.median(confs)), 4),
                "min_confidence": round(float(np.min(confs)), 4),
                "max_confidence": round(float(np.max(confs)), 4)
            }
        else:
            final_class_stats[cls_name] = {
                "detections": 0,
                "images_containing_class": 0,
                "mean_confidence": 0.0,
                "median_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0
            }
            
    # Compile Final Report
    total = global_stats["images_processed"]
    ocr_ready_rate = global_stats["ocr_ready_images"] / total if total > 0 else 0.0
    
    report = {
        "images_processed": total,
        "ocr_ready_images": global_stats["ocr_ready_images"],
        "ocr_ready_rate": round(ocr_ready_rate, 4),
        "images_with_front_panel": global_stats["images_with_front_panel"],
        "images_with_back_panel": global_stats["images_with_back_panel"],
        "images_missing_front_panel": global_stats["images_missing_front_panel"],
        "images_missing_back_panel": global_stats["images_missing_back_panel"],
        "class_statistics": final_class_stats,
        "artwork_statistics": artwork_stats
    }
    
    report_path = os.path.join(stats_dir, "structural_detection_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
        
    print(f"Audit complete! Report saved to {report_path}")

if __name__ == "__main__":
    main()
