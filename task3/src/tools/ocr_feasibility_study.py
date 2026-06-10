import os
import random
import csv
import shutil
import cv2
import re
from paddleocr import PaddleOCR

def ensure_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def main():
    base_dir = r"d:\Silvertouch Internship\task3"
    input_dir = os.path.join(base_dir, "front_back_ocr_clean")
    output_dir = os.path.join(base_dir, "ocr_samples")
    csv_path = os.path.join(base_dir, "ocr_report.csv")
    
    ensure_dir(output_dir)
    
    print("Initializing PaddleOCR...")
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    
    front_dir = os.path.join(input_dir, "Front_Panel")
    back_dir = os.path.join(input_dir, "Back_Panel")
    
    front_files = [os.path.join(front_dir, f) for f in os.listdir(front_dir) if f.endswith('.jpg')]
    back_files = [os.path.join(back_dir, f) for f in os.listdir(back_dir) if f.endswith('.jpg')]
    
    # Sample up to 50
    sample_front = random.sample(front_files, min(50, len(front_files)))
    sample_back = random.sample(back_files, min(50, len(back_files)))
    
    all_samples = [(f, 'Front_Panel') for f in sample_front] + [(f, 'Back_Panel') for f in sample_back]
    
    stats = {
        'total': 0,
        'returned_text': 0,
        'rx_detected': 0,
        'confidences': []
    }
    
    successful_examples = []
    failed_examples = []
    
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "class_name", "detected_text", "ocr_confidence", "contains_rx"])
        
        for idx, (filepath, class_name) in enumerate(all_samples):
            filename = os.path.basename(filepath)
            print(f"Processing {idx+1}/{len(all_samples)}: {filename}")
            
            result = ocr.ocr(filepath)
            
            detected_texts = []
            scores = []
            
            if result and isinstance(result, list) and len(result) > 0:
                res_dict = result[0]
                if 'rec_texts' in res_dict:
                    detected_texts = res_dict['rec_texts']
                if 'rec_scores' in res_dict:
                    scores = res_dict['rec_scores']
                    
            full_text = " ".join(detected_texts)
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            full_text_lower = full_text.lower()
            contains_rx = False
            
            # Simple substring matching + regex word boundary for 'rx'
            if re.search(r'\brx\b', full_text_lower) or 'rx only' in full_text_lower or '℞' in full_text_lower:
                contains_rx = True
                
            stats['total'] += 1
            if detected_texts:
                stats['returned_text'] += 1
                stats['confidences'].append(avg_score)
                
            if contains_rx:
                stats['rx_detected'] += 1
                successful_examples.append(filepath)
            else:
                failed_examples.append(filepath)
                
            writer.writerow([filename, class_name, full_text, round(avg_score, 4), contains_rx])
            
    # Save visual examples
    for i, filepath in enumerate(successful_examples[:10]):
        shutil.copy(filepath, os.path.join(output_dir, f"success_{i}_{os.path.basename(filepath)}"))
        
    for i, filepath in enumerate(failed_examples[:10]):
        shutil.copy(filepath, os.path.join(output_dir, f"failed_{i}_{os.path.basename(filepath)}"))
        
    # Print stats
    avg_overall_conf = sum(stats['confidences']) / len(stats['confidences']) if stats['confidences'] else 0.0
    print("\n--- Summary Statistics ---")
    print(f"Total images tested: {stats['total']}")
    print(f"Images where OCR returned text: {stats['returned_text']}")
    print(f"Images where Rx was detected: {stats['rx_detected']}")
    print(f"Average OCR confidence: {round(avg_overall_conf, 4)}")
    print(f"\nSaved {len(successful_examples[:10])} success examples and {len(failed_examples[:10])} failure examples in {output_dir}")

if __name__ == "__main__":
    main()
