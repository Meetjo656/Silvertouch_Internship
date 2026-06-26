import os
import json
import cv2

JSON_PATH = r"d:\Silvertouch Internship\task3\cluttered_sample\annotations.json"
IMG_DIR = r"d:\Silvertouch Internship\task3\cluttered_sample"
OUT_DIR = r"d:\Silvertouch Internship\task3\cluttered_sample_overlaid"

COLORS = {
    1: (200, 200, 200),  # Artwork - Gray
    2: (150, 150, 150),  # not_artwork - Dark Gray
    3: (0, 255, 0),      # Front - Green
    4: (255, 0, 0),      # Side - Blue
    5: (0, 0, 255),      # Back - Red
    6: (255, 255, 0),    # Extra_Areas - Cyan
    7: (0, 255, 255),    # Bottom - Yellow
    8: (255, 0, 255)     # Top - Magenta
}

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
        
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    
    img_by_id = {img["id"]: img["file_name"] for img in data["images"]}
    anns_by_img = {img["id"]: [] for img in data["images"]}
    for ann in data["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)
        
    for img_id, filename in img_by_id.items():
        img_path = os.path.join(IMG_DIR, filename)
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        for ann in anns_by_img[img_id]:
            x, y, w, h = ann["bbox"]
            cat_id = ann["category_id"]
            cat_name = categories.get(cat_id, "Unknown")
            color = COLORS.get(cat_id, (255, 255, 255))
            
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, cat_name, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        out_path = os.path.join(OUT_DIR, filename)
        cv2.imwrite(out_path, img)
        
    print(f"Processed {len(img_by_id)} images. Saved to {OUT_DIR}")

if __name__ == '__main__':
    main()
