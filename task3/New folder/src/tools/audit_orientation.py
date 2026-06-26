import os
import cv2
import csv
import shutil
from collections import defaultdict
from paddleocr import PaddleOCR

def ensure_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def get_ocr_confidence(ocr, img):
    # cls=False to prevent automatic angle classification, so we can test orientations manually
    result = ocr.ocr(img) 
    if not result or not isinstance(result, list) or len(result) == 0:
        return 0.0
    res_dict = result[0]
    if 'rec_scores' not in res_dict:
        return 0.0
    scores = res_dict['rec_scores']
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

def classify_orientation(conf_orig, conf_rot180, conf_mirror, conf_m_rot180):
    best_normal = max(conf_orig, conf_rot180)
    best_mirror = max(conf_mirror, conf_m_rot180)
    
    if best_mirror > best_normal + 0.1 and best_mirror > 0.5:
        return "MIRRORED", "REJECT"
        
    if best_normal < 0.4:
        return "UNCERTAIN", "REJECT"
        
    if conf_rot180 > conf_orig * 1.1:
        return "ROTATED_180", "REJECT"
    elif conf_orig > conf_rot180 * 1.1:
        return "UPRIGHT", "KEEP"
    else:
        if conf_orig > 0.6:
            return "UPRIGHT", "KEEP"
        else:
            return "UNCERTAIN", "REJECT"

def main():
    base_dir = r"d:\Silvertouch Internship\task3"
    input_dir = os.path.join(base_dir, "front_back_clean")
    audit_dir = os.path.join(base_dir, "orientation_audit")
    csv_path = os.path.join(base_dir, "orientation_audit.csv")
    final_dir = os.path.join(base_dir, "front_back_ocr_clean")
    
    ensure_dir(audit_dir)
    for cat in ["upright", "rotated_180", "mirrored", "uncertain"]:
        os.makedirs(os.path.join(audit_dir, cat))
        
    ensure_dir(final_dir)
    os.makedirs(os.path.join(final_dir, "Front_Panel"))
    os.makedirs(os.path.join(final_dir, "Back_Panel"))
    
    print("Initializing PaddleOCR...")
    ocr = PaddleOCR(use_textline_orientation=False, lang='en')
    
    stats = defaultdict(int)
    class_stats = defaultdict(int)
    samples_saved = defaultdict(int)
    
    all_files = []
    for cls_name in ["Front_Panel", "Back_Panel"]:
        cls_dir = os.path.join(input_dir, cls_name)
        if not os.path.exists(cls_dir): continue
        for f in os.listdir(cls_dir):
            if f.endswith(('.jpg', '.png')):
                all_files.append((cls_name, os.path.join(cls_dir, f)))
                class_stats[cls_name] += 1
                
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "class_name", "width", "height", "original_confidence", "rotated_confidence", "best_orientation", "status"])
        
        for idx, (cls_name, filepath) in enumerate(all_files):
            filename = os.path.basename(filepath)
            print(f"Processing {idx+1}/{len(all_files)}: {filename}")
            
            img = cv2.imread(filepath)
            if img is None:
                continue
                
            h, w = img.shape[:2]
            
            img_rot180 = cv2.rotate(img, cv2.ROTATE_180)
            img_mirror = cv2.flip(img, 1)
            img_mirror_rot180 = cv2.rotate(img_mirror, cv2.ROTATE_180)
            
            conf_orig = get_ocr_confidence(ocr, img)
            conf_rot180 = get_ocr_confidence(ocr, img_rot180)
            conf_mirror = get_ocr_confidence(ocr, img_mirror)
            conf_m_rot180 = get_ocr_confidence(ocr, img_mirror_rot180)
            
            best_orientation, status = classify_orientation(conf_orig, conf_rot180, conf_mirror, conf_m_rot180)
            stats[best_orientation] += 1
            
            writer.writerow([filename, cls_name, w, h, round(conf_orig, 4), round(conf_rot180, 4), best_orientation, status])
            
            target_cat = best_orientation.lower()
            if samples_saved[target_cat] < 20:
                shutil.copy(filepath, os.path.join(audit_dir, target_cat, filename))
                samples_saved[target_cat] += 1
                
            if status == "KEEP":
                shutil.copy(filepath, os.path.join(final_dir, cls_name, filename))
                
    print("\n--- Orientation Audit Statistics ---")
    print(f"Total Front_Panel crops: {class_stats['Front_Panel']}")
    print(f"Total Back_Panel crops: {class_stats['Back_Panel']}")
    
    total = len(all_files)
    if total > 0:
        print(f"\nUPRIGHT: {stats['UPRIGHT']} ({stats['UPRIGHT']/total*100:.1f}%)")
        print(f"ROTATED_180: {stats['ROTATED_180']} ({stats['ROTATED_180']/total*100:.1f}%)")
        print(f"MIRRORED: {stats['MIRRORED']} ({stats['MIRRORED']/total*100:.1f}%)")
        print(f"UNCERTAIN: {stats['UNCERTAIN']} ({stats['UNCERTAIN']/total*100:.1f}%)")

if __name__ == "__main__":
    main()
