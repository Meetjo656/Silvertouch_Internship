import os
import cv2
import json
import re
from pathlib import Path
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def is_rx_match(text):
    text = text.strip()
    # Case insensitive match for rx, rx only, rxonly, or the Rx symbol
    pattern = r'\b(rx\s*only|rxonly|rx)\b|\u211e'
    return bool(re.search(pattern, text, re.IGNORECASE))

def main():
    base_dir = r"d:\Silvertouch Internship\task3"
    model_path = os.path.join(base_dir, r"trained_models\structural_8class_20epochs\weights\20_epochs_8classes.pt")
    input_dirs = [
        os.path.join(base_dir, "front_back_clean", "Front_Panel"),
        os.path.join(base_dir, "front_back_clean", "Back_Panel")
    ]
    
    panel_crops_dir = os.path.join(base_dir, "panel_crops")
    annotated_dir = os.path.join(base_dir, "outputs", "annotated")
    results_json_path = os.path.join(base_dir, "results.json")
    evaluation_json_path = os.path.join(base_dir, "evaluation_report.json")
    
    ensure_dir(panel_crops_dir)
    ensure_dir(annotated_dir)
    
    print("Loading YOLO model...")
    yolo_model = YOLO(model_path)
    class_names = yolo_model.names
    
    print("Loading PaddleOCR...")
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    
    all_images = []
    for d in input_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append(os.path.join(d, f))
                    
    print(f"Found {len(all_images)} images for stress testing.")
    
    eval_stats = {
        "images_processed": 0,
        "yolo_success_count": 0,
        "ocr_success_count": 0,
        "rx_found_count": 0,
        "rx_not_found_count": 0,
        "ocr_uncertain_count": 0
    }
    
    all_results = []
    
    for idx, img_path in enumerate(all_images):
        filename = os.path.basename(img_path)
        print(f"[{idx+1}/{len(all_images)}] Processing: {filename}")
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        annotated_img = img.copy()
        
        # Phase 1: YOLO Detection
        results = yolo_model(img, verbose=False)
        
        front_detected = False
        back_detected = False
        
        panels_to_process = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = class_names[cls_id]
                conf = float(box.conf[0].item())
                
                if cls_name in ["Front_Panel", "Back_Panel"]:
                    if cls_name == "Front_Panel": front_detected = True
                    if cls_name == "Back_Panel": back_detected = True
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Phase 2: Cropping
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0: continue
                    
                    crop_filename = f"{cls_name}_{filename}"
                    cv2.imwrite(os.path.join(panel_crops_dir, crop_filename), crop)
                    
                    panels_to_process.append({
                        "class": cls_name,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "crop": crop,
                        "offset_x": x1,
                        "offset_y": y1
                    })
                    
                    # Draw YOLO boxes
                    color = (255, 0, 0) if cls_name == "Front_Panel" else (0, 0, 255)
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_img, cls_name, (x1, max(10, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        yolo_success = front_detected or back_detected
        if yolo_success:
            eval_stats["yolo_success_count"] += 1
            
        # Phase 3: OCR Processing
        rx_matches = []
        ocr_all_results = []
        
        total_ocr_lines = 0
        total_ocr_conf = 0.0
        
        for panel in panels_to_process:
            crop = panel["crop"]
            offset_x = panel["offset_x"]
            offset_y = panel["offset_y"]
            
            ocr_res = ocr.ocr(crop)
            
            if ocr_res and isinstance(ocr_res, list) and len(ocr_res) > 0 and ocr_res[0]:
                res_dict = ocr_res[0]
                if 'rec_texts' in res_dict and 'rec_scores' in res_dict and 'dt_polys' in res_dict:
                    texts = res_dict['rec_texts']
                    scores = res_dict['rec_scores']
                    boxes = res_dict['dt_polys']
                    
                    for txt, score, box in zip(texts, scores, boxes):
                        total_ocr_lines += 1
                        total_ocr_conf += score
                        
                        ocr_all_results.append({
                            "text": txt,
                            "confidence": round(float(score), 4)
                        })
                        
                        # Phase 4: Rx Detection
                        if is_rx_match(txt):
                            # Phase 5: Localization
                            abs_box = []
                            for pt in box:
                                abs_x = int(pt[0]) + offset_x
                                abs_y = int(pt[1]) + offset_y
                                abs_box.append([abs_x, abs_y])
                                
                            rx_matches.append({
                                "text": txt,
                                "confidence": round(float(score), 4),
                                "bbox": abs_box
                            })
                            
                            # Phase 6: Visualization
                            pts = np.array(abs_box, np.int32)
                            pts = pts.reshape((-1, 1, 2))
                            cv2.polylines(annotated_img, [pts], True, (0, 255, 0), 3)
                            cv2.putText(annotated_img, "Rx Found", (abs_box[0][0], max(10, abs_box[0][1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Phase 3.5: OCR Quality Validation
        avg_ocr_conf = total_ocr_conf / total_ocr_lines if total_ocr_lines > 0 else 0.0
        
        # Phase 8: Warning Logic
        rx_found = len(rx_matches) > 0
        
        # Only run OCR validation if we actually processed some panels
        if not yolo_success:
            status = "OCR_UNCERTAIN"
            eval_stats["ocr_uncertain_count"] += 1
        elif total_ocr_lines < 2 or avg_ocr_conf < 0.50:
            status = "OCR_UNCERTAIN"
            eval_stats["ocr_uncertain_count"] += 1
        else:
            eval_stats["ocr_success_count"] += 1
            if rx_found:
                status = "RX_FOUND"
                eval_stats["rx_found_count"] += 1
            else:
                status = "RX_NOT_FOUND"
                eval_stats["rx_not_found_count"] += 1
                
        eval_stats["images_processed"] += 1
        
        # Phase 6: Save Annotated Image
        cv2.imwrite(os.path.join(annotated_dir, filename), annotated_img)
        
        # Phase 7: JSON Report
        result_json = {
            "image": filename,
            "front_panel_detected": front_detected,
            "back_panel_detected": back_detected,
            "rx_found": rx_found,
            "status": status,
            "rx_matches": rx_matches,
            "ocr_results": ocr_all_results
        }
        all_results.append(result_json)
        
    # Generate overall JSON Reports
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
        
    total = eval_stats["images_processed"]
    if total > 0:
        eval_report = {
            "images_processed": total,
            "yolo_success_rate": round(eval_stats["yolo_success_count"] / total, 4),
            "ocr_success_rate": round(eval_stats["ocr_success_count"] / total, 4),
            "rx_found_count": eval_stats["rx_found_count"],
            "rx_not_found_count": eval_stats["rx_not_found_count"],
            "ocr_uncertain_count": eval_stats["ocr_uncertain_count"]
        }
        with open(evaluation_json_path, 'w', encoding='utf-8') as f:
            json.dump(eval_report, f, indent=4)
            
    print("Pipeline Execution Complete!")
    print(f"Results saved to {results_json_path}")
    print(f"Evaluation report saved to {evaluation_json_path}")

if __name__ == "__main__":
    main()
