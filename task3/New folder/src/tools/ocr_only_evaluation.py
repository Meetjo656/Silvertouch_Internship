import os
import cv2
import json
import re
import shutil
import numpy as np
from pathlib import Path
from paddleocr import PaddleOCR

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def normalize_text(text):
    # Lowercase, replace multiple spaces with single space, strip
    t = text.lower()
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

def find_rx_matches(norm_text):
    # Regex for rx, rx only, rxonly, or the ℞ symbol
    pattern = r'\b(rx\s*only|rxonly|rx)\b|\u211e'
    # Return all matches found in the normalized text
    return re.findall(pattern, norm_text)

def main():
    base_dir = r"d:\Silvertouch Internship\task3"
    input_base = os.path.join(base_dir, "front_back_clean")
    
    dirs = {
        "Front_Panel": os.path.join(input_base, "Front_Panel"),
        "Back_Panel": os.path.join(input_base, "Back_Panel")
    }
    
    uncertain_dir = os.path.join(base_dir, "ocr_uncertain_samples")
    report_json_path = os.path.join(base_dir, "ocr_only_evaluation.json")
    
    ensure_dir(uncertain_dir)
    
    print("Loading PaddleOCR...")
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    
    all_images = []
    for panel_type, d in dirs.items():
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append({
                        "path": os.path.join(d, f),
                        "filename": f,
                        "panel_type": panel_type
                    })
                    
    print(f"Found {len(all_images)} images for OCR evaluation.")
    
    global_stats = {
        "images_processed": 0,
        "ocr_success_count": 0,
        "rx_found_count": 0,
        "rx_not_found_count": 0,
        "ocr_uncertain_count": 0,
        "all_confidences": []
    }
    
    panel_stats = {
        "Front_Panel": {"images": 0, "ocr_success": 0, "ocr_uncertain": 0, "rx_found": 0, "rx_not_found": 0},
        "Back_Panel": {"images": 0, "ocr_success": 0, "ocr_uncertain": 0, "rx_found": 0, "rx_not_found": 0}
    }
    
    detailed_results = []
    successful_images_for_sorting = []
    
    for idx, item in enumerate(all_images):
        img_path = item["path"]
        filename = item["filename"]
        p_type = item["panel_type"]
        
        print(f"[{idx+1}/{len(all_images)}] Processing: {filename} ({p_type})")
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        global_stats["images_processed"] += 1
        panel_stats[p_type]["images"] += 1
        
        ocr_res = ocr.ocr(img)
        
        total_words = 0
        total_chars = 0
        total_conf = 0.0
        line_count = 0
        raw_texts = []
        
        raw_ocr_results = []
        
        if ocr_res and isinstance(ocr_res, list) and len(ocr_res) > 0 and ocr_res[0]:
            res_dict = ocr_res[0]
            if 'rec_texts' in res_dict and 'rec_scores' in res_dict and 'dt_polys' in res_dict:
                texts = res_dict['rec_texts']
                scores = res_dict['rec_scores']
                boxes = res_dict['dt_polys']
                
                for txt, score, box in zip(texts, scores, boxes):
                    words = len(txt.split())
                    chars = len(txt.replace(" ", ""))
                    
                    total_words += words
                    total_chars += chars
                    total_conf += score
                    line_count += 1
                    
                    raw_texts.append(txt)
                    raw_ocr_results.append({
                        "text": txt,
                        "confidence": round(float(score), 4),
                        "bbox": [[int(pt[0]), int(pt[1])] for pt in box]
                    })
                    
        avg_conf = total_conf / line_count if line_count > 0 else 0.0
        
        raw_ocr_text = "\n".join(raw_texts)
        normalized_text = normalize_text(raw_ocr_text)
        
        is_uncertain = False
        if avg_conf < 0.50 or total_words == 0 or total_chars == 0:
            is_uncertain = True
            
        rx_matches = []
        rx_found = False
        status = ""
        
        if is_uncertain:
            status = "OCR_UNCERTAIN"
            global_stats["ocr_uncertain_count"] += 1
            panel_stats[p_type]["ocr_uncertain"] += 1
            
            # Copy to uncertain dir
            dest_path = os.path.join(uncertain_dir, filename)
            shutil.copy2(img_path, dest_path)
        else:
            global_stats["ocr_success_count"] += 1
            panel_stats[p_type]["ocr_success"] += 1
            global_stats["all_confidences"].append(avg_conf)
            
            rx_matches = find_rx_matches(normalized_text)
            rx_found = len(rx_matches) > 0
            
            if rx_found:
                status = "RX_FOUND"
                global_stats["rx_found_count"] += 1
                panel_stats[p_type]["rx_found"] += 1
            else:
                status = "RX_NOT_FOUND"
                global_stats["rx_not_found_count"] += 1
                panel_stats[p_type]["rx_not_found"] += 1
                
            successful_images_for_sorting.append({
                "filename": filename,
                "panel_type": p_type,
                "confidence": round(avg_conf, 4),
                "status": status,
                "rx_found": rx_found
            })
            
        image_report = {
            "image": filename,
            "panel_type": p_type,
            "status": status,
            "ocr_quality": {
                "words": total_words,
                "characters": total_chars,
                "avg_confidence": round(avg_conf, 4)
            },
            "raw_ocr_text": raw_ocr_text,
            "normalized_text": normalized_text,
            "rx_found": rx_found,
            "rx_matches": rx_matches,
            "ocr_results": raw_ocr_results
        }
        
        detailed_results.append(image_report)
        
    # Phase 8 & 9: Confidence Analysis & Extreme Samples
    all_confs = global_stats["all_confidences"]
    if len(all_confs) > 0:
        avg_confidence = float(np.mean(all_confs))
        median_confidence = float(np.median(all_confs))
        min_confidence = float(np.min(all_confs))
        max_confidence = float(np.max(all_confs))
    else:
        avg_confidence = median_confidence = min_confidence = max_confidence = 0.0
        
    # Sort successful images by confidence descending
    successful_images_for_sorting.sort(key=lambda x: x["confidence"], reverse=True)
    
    top_20 = successful_images_for_sorting[:20]
    bottom_20 = successful_images_for_sorting[-20:] if len(successful_images_for_sorting) >= 20 else successful_images_for_sorting

    total = global_stats["images_processed"]
    ocr_success_rate = global_stats["ocr_success_count"] / total if total > 0 else 0.0
    
    for pt in ["Front_Panel", "Back_Panel"]:
        p_total = panel_stats[pt]["images"]
        if p_total > 0:
            panel_stats[pt]["ocr_success_rate"] = round(panel_stats[pt]["ocr_success"] / p_total, 4)
        else:
            panel_stats[pt]["ocr_success_rate"] = 0.0
            
    final_report = {
        "images_processed": total,
        "ocr_success_rate": round(ocr_success_rate, 4),
        "rx_found_count": global_stats["rx_found_count"],
        "rx_not_found_count": global_stats["rx_not_found_count"],
        "ocr_uncertain_count": global_stats["ocr_uncertain_count"],
        "average_confidence": round(avg_confidence, 4),
        "median_confidence": round(median_confidence, 4),
        "min_confidence": round(min_confidence, 4),
        "max_confidence": round(max_confidence, 4),
        "front_panel_statistics": panel_stats["Front_Panel"],
        "back_panel_statistics": panel_stats["Back_Panel"],
        "highest_confidence_examples": top_20,
        "lowest_confidence_examples": bottom_20,
        "detailed_results": detailed_results
    }
    
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4)
        
    print(f"Evaluation complete! Results saved to {report_json_path}")
    print(f"Uncertain samples copied to {uncertain_dir}")

if __name__ == "__main__":
    main()
