import os
import cv2
import json
import re
import shutil
import numpy as np
from paddleocr import PaddleOCR
import time

def ensure_dir(d):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

def is_rx_match(text):
    # Normalize whitespace
    text = " ".join(text.split())
    # Regex to match exactly rx, rx only, rxonly, or ℞ with optional surrounding punctuation.
    # We use word boundaries for the english letters so we don't match "prix" or "borax".
    pattern = re.compile(r'\b(rx\s*only|rxonly|rx)\b|℞', re.IGNORECASE)
    return bool(pattern.search(text))

def main():
    base_dir = r"d:\Silvertouch Internship\task3"
    input_dir = os.path.join(base_dir, "outputs", "all_panels_cropped")
    
    outputs_dir = os.path.join(base_dir, "outputs")
    overlays_dir = os.path.join(outputs_dir, "rx_overlays")
    rx_found_dir = os.path.join(outputs_dir, "rx_found")
    rx_not_found_dir = os.path.join(outputs_dir, "rx_not_found")
    
    ensure_dir(overlays_dir)
    ensure_dir(rx_found_dir)
    ensure_dir(rx_not_found_dir)
    
    # Initialize PaddleOCR
    print("Initializing PaddleOCR...")
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    
    # Locate all images
    all_images = []
    if os.path.exists(input_dir):
        for root, dirs, files in os.walk(input_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    panel_type = os.path.basename(root)
                    all_images.append({
                        "path": os.path.join(root, f),
                        "filename": f,
                        "panel_type": panel_type
                    })
                    
    print(f"Found {len(all_images)} images to process.")
    
    results = []
    
    summary = {
        "images_processed": 0,
        "front_panel_images": 0,
        "back_panel_images": 0,
        "side_panel_images": 0,
        "top_panel_images": 0,
        "bottom_panel_images": 0,
        "total_rx_found": 0,
        "images_with_rx": 0,
        "images_without_rx": 0,
        "average_ocr_confidence": 0.0
    }
    
    all_confidences = []
    
    for idx, img_data in enumerate(all_images):
        img_path = img_data["path"]
        filename = img_data["filename"]
        panel_type = img_data["panel_type"]
        
        print(f"[{idx+1}/{len(all_images)}] Processing {panel_type}: {filename}")
        
        # Update metrics
        summary["images_processed"] += 1
        if panel_type.lower() == "front_panel": summary["front_panel_images"] += 1
        elif panel_type.lower() == "back_panel": summary["back_panel_images"] += 1
        elif panel_type.lower() == "side_panel": summary["side_panel_images"] += 1
        elif panel_type.lower() == "top_panel": summary["top_panel_images"] += 1
        elif panel_type.lower() == "bottom_panel": summary["bottom_panel_images"] += 1
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        overlay_img = img.copy()
        
        # Run OCR using new predict API
        try:
            ocr_res_list = list(ocr.predict(img))
        except Exception as e:
            print(f"OCR Error: {e}")
            continue
            
        raw_texts = []
        rx_matches = []
        
        if ocr_res_list and len(ocr_res_list) > 0:
            res_dict = ocr_res_list[0]
            if 'rec_texts' in res_dict and 'dt_polys' in res_dict and 'rec_scores' in res_dict:
                texts = res_dict['rec_texts']
                scores = res_dict['rec_scores']
                polys = res_dict['dt_polys']
                
                for i in range(len(texts)):
                    text = texts[i]
                    conf = float(scores[i])
                    # dt_polys is list of arrays, convert to list of lists of floats for JSON
                    bbox_np = polys[i]
                    bbox = [[float(pt[0]), float(pt[1])] for pt in bbox_np]
                    
                    raw_texts.append(text)
                    all_confidences.append(conf)
                    
                    is_rx = is_rx_match(text)
                    
                    # Draw boxes
                    pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
                
                if is_rx:
                    # Highlight Rx matches in Red
                    cv2.polylines(overlay_img, [pts], True, (0, 0, 255), 3)
                    rx_matches.append({
                        "text": text,
                        "confidence": round(conf, 4),
                        "bbox": bbox
                    })
                    
                    rx_idx = len(rx_matches)
                    # Put label above bounding box
                    x, y = int(bbox[0][0]), int(bbox[0][1])
                    cv2.putText(overlay_img, f"Rx #{rx_idx}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                else:
                    # Normal OCR in Green
                    cv2.polylines(overlay_img, [pts], True, (0, 255, 0), 1)
        
        raw_ocr_text = "\n".join(raw_texts)
        normalized_ocr_text = " ".join(raw_texts).lower()
        normalized_ocr_text = " ".join(normalized_ocr_text.split())
        
        rx_count = len(rx_matches)
        
        results.append({
            "image_name": f"{panel_type}_{filename}",
            "panel_type": panel_type,
            "rx_count": rx_count,
            "rx_found": rx_count > 0,
            "rx_matches": rx_matches,
            "raw_ocr_text": raw_ocr_text,
            "normalized_ocr_text": normalized_ocr_text
        })
        
        # Route file
        if rx_count > 0:
            shutil.copy(img_path, os.path.join(rx_found_dir, f"{panel_type}_{filename}"))
            summary["images_with_rx"] += 1
            summary["total_rx_found"] += rx_count
        else:
            shutil.copy(img_path, os.path.join(rx_not_found_dir, f"{panel_type}_{filename}"))
            summary["images_without_rx"] += 1
            
        # Save overlay
        cv2.imwrite(os.path.join(overlays_dir, f"{panel_type}_{filename}"), overlay_img)

    if len(all_confidences) > 0:
        summary["average_ocr_confidence"] = round(float(np.mean(all_confidences)), 4)
        
    print("Writing reports...")
    
    with open(os.path.join(base_dir, "rx_detection_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    with open(os.path.join(base_dir, "rx_summary_report.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
        
    print("Rx Localization Pipeline Validation Complete!")

if __name__ == "__main__":
    main()
