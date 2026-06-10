import os
import cv2
import json
import shutil
import csv

def ensure_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def get_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score

def process_coco_dir(coco_dir, output_dir, csv_writer, stats, config):
    anno_path = os.path.join(coco_dir, "_annotations.coco.json")
    if not os.path.exists(anno_path):
        print(f"Skipping {coco_dir}, no COCO annotations found.")
        return
        
    with open(anno_path, 'r') as f:
        data = json.load(f)
        
    categories = data.get('categories', [])
    valid_ids = {}
    for c in categories:
        name = c['name'].lower()
        if 'front' in name:
            valid_ids[c['id']] = 'Front_Panel'
        elif 'back' in name:
            valid_ids[c['id']] = 'Back_Panel'
            
    img_to_annos = {}
    for ann in data.get('annotations', []):
        if ann['category_id'] in valid_ids:
            img_id = ann['image_id']
            if img_id not in img_to_annos:
                img_to_annos[img_id] = []
            img_to_annos[img_id].append(ann)
            
    for img_info in data.get('images', []):
        img_id = img_info['id']
        annos = img_to_annos.get(img_id, [])
        if not annos:
            continue
            
        file_name = img_info['file_name']
        img_path = os.path.join(coco_dir, file_name)
        
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_h, img_w = img.shape[:2]
        
        for i, ann in enumerate(annos):
            cat_name = valid_ids[ann['category_id']]
            if cat_name == 'Front_Panel':
                stats['total_front'] += 1
            else:
                stats['total_back'] += 1
                
            bbox = ann['bbox'] # [x_min, y_min, width, height]
            x, y, w, h = [int(v) for v in bbox]
            
            # 15% padding
            pad_w = int(w * config['padding_pct'])
            pad_h = int(h * config['padding_pct'])
            
            x_min = max(0, x - pad_w)
            y_min = max(0, y - pad_h)
            x_max = min(img_w, x + w + pad_w)
            y_max = min(img_h, y + h + pad_h)
            
            crop_w = x_max - x_min
            crop_h = y_max - y_min
            
            if crop_w <= 0 or crop_h <= 0:
                continue
                
            crop = img[y_min:y_max, x_min:x_max]
            
            base_name = os.path.splitext(file_name)[0]
            out_name = f"{cat_name}_{base_name}_crop{i}.jpg"
            
            pixels = crop_w * crop_h
            sharpness = get_sharpness(crop)
            
            if crop_w < config['min_width'] or crop_h < config['min_height']:
                status = "REJECTED_LOW_RESOLUTION"
                stats['rejected_resolution'] += 1
            elif sharpness < config['min_sharpness']:
                status = "REJECTED_LOW_SHARPNESS"
                stats['rejected_sharpness'] += 1
            else:
                status = "KEPT"
                stats['kept'] += 1
                out_path = os.path.join(output_dir, cat_name, out_name)
                cv2.imwrite(out_path, crop)
                
            csv_writer.writerow([out_name, cat_name, crop_w, crop_h, pixels, round(sharpness, 2), status])

def main():
    base_dir = r"d:\Silvertouch Internship\task3"
    output_dir = os.path.join(base_dir, "front_back_clean")
    
    # Configuration
    config = {
        'padding_pct': 0.15,
        'min_width': 0,
        'min_height': 0,
        'min_sharpness': 0
    }
    
    print(f"Recreating clean dataset at {output_dir}...")
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "Front_Panel"))
    ensure_dir(os.path.join(output_dir, "Back_Panel"))
    
    csv_path = os.path.join(output_dir, "dataset_report.csv")
    
    stats = {
        'total_front': 0,
        'total_back': 0,
        'rejected_resolution': 0,
        'rejected_sharpness': 0,
        'kept': 0
    }
    
    coco_dirs = [
        os.path.join(base_dir, "train"),
        os.path.join(base_dir, "test"),
        os.path.join(base_dir, "valid")
    ]
    
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "class_name", "width", "height", "pixels", "sharpness_score", "status"])
        
        for cdir in coco_dirs:
            print(f"Processing COCO directory: {cdir}")
            process_coco_dir(cdir, output_dir, writer, stats, config)
            
    print("\n--- Final Statistics ---")
    print(f"Total Front_Panel crops: {stats['total_front']}")
    print(f"Total Back_Panel crops: {stats['total_back']}")
    print(f"Rejected for low resolution: {stats['rejected_resolution']}")
    print(f"Rejected for low sharpness: {stats['rejected_sharpness']}")
    print(f"Final usable images (KEPT): {stats['kept']}")
    print(f"Report saved to: {csv_path}")

if __name__ == "__main__":
    main()
