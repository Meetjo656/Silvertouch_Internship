import os
import cv2
import json
import csv
import random
import statistics
import shutil

def ensure_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def get_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score

def main():
    base_dir = r"d:\Silvertouch Internship\task3"
    output_dir = os.path.join(base_dir, "audit_samples")
    csv_path = os.path.join(base_dir, "crop_statistics.csv")
    
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "Front_Panel"))
    ensure_dir(os.path.join(output_dir, "Back_Panel"))
    
    splits = ["train", "valid", "test"]
    
    all_categories = set()
    
    coverage = {
        'train': {'images': 0, 'Front_Panel': 0, 'Back_Panel': 0},
        'valid': {'images': 0, 'Front_Panel': 0, 'Back_Panel': 0},
        'test': {'images': 0, 'Front_Panel': 0, 'Back_Panel': 0},
        'total': {'images': 0, 'Front_Panel': 0, 'Back_Panel': 0}
    }
    
    # Storage for statistics
    stats = {
        'Front_Panel': {'widths': [], 'heights': [], 'sharpness': []},
        'Back_Panel': {'widths': [], 'heights': [], 'sharpness': []}
    }
    
    # Store all crops to randomly select later
    saved_crops_front = []
    saved_crops_back = []
    
    # First, let's discover categories
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        # Assuming we should use _annotations_mapped.json based on class names requested
        anno_path = os.path.join(split_dir, "_annotations_mapped.json")
        if not os.path.exists(anno_path):
            anno_path = os.path.join(split_dir, "_annotations.coco.json")
            if not os.path.exists(anno_path):
                continue
                
        with open(anno_path, 'r') as f:
            data = json.load(f)
            
        for c in data.get('categories', []):
            all_categories.add(c['name'])
            
    print("Found Categories:")
    for cat in sorted(list(all_categories)):
        print(f"* {cat}")
        
    print("\nSelected Categories:")
    print("* Front_Panel")
    print("* Back_Panel\n")
    
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "class_name", "source_split", "width", "height", "pixels", "sharpness_score"])
        
        for split in splits:
            split_dir = os.path.join(base_dir, split)
            anno_path = os.path.join(split_dir, "_annotations_mapped.json")
            if not os.path.exists(anno_path):
                anno_path = os.path.join(split_dir, "_annotations.coco.json")
                if not os.path.exists(anno_path):
                    continue
                    
            with open(anno_path, 'r') as f:
                data = json.load(f)
                
            categories = data.get('categories', [])
            cat_id_to_name = {c['id']: c['name'] for c in categories}
            
            # Map exactly 'Front_Panel' and 'Back_Panel'
            valid_ids = {}
            for c in categories:
                if c['name'] == 'Front_Panel':
                    valid_ids[c['id']] = 'Front_Panel'
                elif c['name'] == 'Back_Panel':
                    valid_ids[c['id']] = 'Back_Panel'
                    
            coverage[split]['images'] = len(data.get('images', []))
            coverage['total']['images'] += len(data.get('images', []))
            
            img_to_annos = {}
            for ann in data.get('annotations', []):
                cat_id = ann['category_id']
                if cat_id in valid_ids:
                    cat_name = valid_ids[cat_id]
                    coverage[split][cat_name] += 1
                    coverage['total'][cat_name] += 1
                    
                    img_id = ann['image_id']
                    if img_id not in img_to_annos:
                        img_to_annos[img_id] = []
                    img_to_annos[img_id].append(ann)
                    
            # Process images for crops
            for img_info in data.get('images', []):
                img_id = img_info['id']
                annos = img_to_annos.get(img_id, [])
                if not annos:
                    continue
                    
                file_name = img_info['file_name']
                img_path = os.path.join(split_dir, file_name)
                if not os.path.exists(img_path):
                    continue
                    
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                img_h, img_w = img.shape[:2]
                
                for i, ann in enumerate(annos):
                    cat_name = valid_ids[ann['category_id']]
                    bbox = ann['bbox'] # [x_min, y_min, w, h]
                    x, y, w, h = [int(v) for v in bbox]
                    
                    # 15% padding
                    pad_w = int(w * 0.15)
                    pad_h = int(h * 0.15)
                    
                    x_min = max(0, x - pad_w)
                    y_min = max(0, y - pad_h)
                    x_max = min(img_w, x + w + pad_w)
                    y_max = min(img_h, y + h + pad_h)
                    
                    crop_w = x_max - x_min
                    crop_h = y_max - y_min
                    
                    if crop_w <= 0 or crop_h <= 0:
                        continue
                        
                    crop = img[y_min:y_max, x_min:x_max]
                    pixels = crop_w * crop_h
                    sharpness = get_sharpness(crop)
                    
                    base_name = os.path.splitext(file_name)[0]
                    out_name = f"{cat_name}_{base_name}_crop{i}.jpg"
                    
                    stats[cat_name]['widths'].append(crop_w)
                    stats[cat_name]['heights'].append(crop_h)
                    stats[cat_name]['sharpness'].append(sharpness)
                    
                    writer.writerow([out_name, cat_name, split, crop_w, crop_h, pixels, round(sharpness, 2)])
                    
                    # Store temporarily to sample later
                    if cat_name == 'Front_Panel':
                        saved_crops_front.append((out_name, crop))
                    else:
                        saved_crops_back.append((out_name, crop))

    # Print Dataset Coverage
    for split in splits:
        print(f"{split.upper()}")
        print(f"Images: {coverage[split]['images']}")
        print(f"Front_Panel: {coverage[split]['Front_Panel']}")
        print(f"Back_Panel : {coverage[split]['Back_Panel']}\n")
        
    print(f"TOTAL")
    print(f"Images: {coverage['total']['images']}")
    print(f"Front_Panel: {coverage['total']['Front_Panel']}")
    print(f"Back_Panel : {coverage['total']['Back_Panel']}\n")
    
    # Save audit samples
    print("Saving audit samples...")
    sample_front = random.sample(saved_crops_front, min(20, len(saved_crops_front)))
    sample_back = random.sample(saved_crops_back, min(20, len(saved_crops_back)))
    
    for name, crop in sample_front:
        cv2.imwrite(os.path.join(output_dir, "Front_Panel", name), crop)
        
    for name, crop in sample_back:
        cv2.imwrite(os.path.join(output_dir, "Back_Panel", name), crop)
        
    # Print Distribution Analysis
    for cat in ['Front_Panel', 'Back_Panel']:
        print(cat)
        w_list = sorted(stats[cat]['widths'])
        h_list = sorted(stats[cat]['heights'])
        s_list = sorted(stats[cat]['sharpness'])
        
        if not w_list:
            print("No data.\n")
            continue
            
        print("Width:")
        print(f"Min: {min(w_list)}")
        print(f"Median: {statistics.median(w_list)}")
        print(f"Max: {max(w_list)}\n")
        
        print("Height:")
        print(f"Min: {min(h_list)}")
        print(f"Median: {statistics.median(h_list)}")
        print(f"Max: {max(h_list)}\n")
        
        print("Sharpness:")
        print(f"Min: {round(min(s_list), 2)}")
        print(f"Median: {round(statistics.median(s_list), 2)}")
        print(f"Max: {round(max(s_list), 2)}\n")

if __name__ == "__main__":
    main()
