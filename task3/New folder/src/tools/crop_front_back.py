import os
import cv2
from glob import glob

def yolo_to_pixel(bbox, img_w, img_h):
    # bbox: [x_center, y_center, width, height] normalized
    x_c, y_c, w_norm, h_norm = bbox
    x1 = int((x_c - w_norm/2) * img_w)
    y1 = int((y_c - h_norm/2) * img_h)
    x2 = int((x_c + w_norm/2) * img_w)
    y2 = int((y_c + h_norm/2) * img_h)
    
    # Ensure within bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    return x1, y1, x2, y2

def main():
    base_dir = r"d:\Silvertouch Internship\task3"
    train_images_dir = os.path.join(base_dir, r"yolo_detection_dataset\train\images")
    train_labels_dir = os.path.join(base_dir, r"yolo_detection_dataset\train\labels")
    output_dir = os.path.join(base_dir, r"front&back")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # In the 8-class taxonomy:
    # 2: Front_Panel
    # 4: Back_Panel
    class_names = {2: "Front_Panel", 4: "Back_Panel"}
    
    label_files = glob(os.path.join(train_labels_dir, "*.txt"))
    
    count = 0
    for txt_path in label_files:
        basename = os.path.basename(txt_path)
        img_name = basename.replace(".txt", ".jpg")
        img_path = os.path.join(train_images_dir, img_name)
        
        if not os.path.exists(img_path):
            img_name = basename.replace(".txt", ".png")
            img_path = os.path.join(train_images_dir, img_name)
            if not os.path.exists(img_path):
                continue
                
        # Read labels
        with open(txt_path, "r") as f:
            lines = f.readlines()
            
        panels_to_crop = []
        for line in lines:
            parts = line.strip().split()
            if parts:
                cls_id = int(parts[0])
                if cls_id in class_names:
                    bbox = [float(x) for x in parts[1:5]]
                    panels_to_crop.append((cls_id, bbox))
                    
        if not panels_to_crop:
            continue
            
        # Load image only if we need to crop
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        for idx, (cls_id, bbox) in enumerate(panels_to_crop):
            x1, y1, x2, y2 = yolo_to_pixel(bbox, w, h)
            
            # Skip invalid crops
            if x2 <= x1 or y2 <= y1:
                continue
                
            cropped = img[y1:y2, x1:x2]
            
            # Save cropped image
            label_name = class_names[cls_id]
            out_name = f"{label_name}_{idx}_{os.path.basename(img_path)}"
            out_path = os.path.join(output_dir, out_name)
            
            cv2.imwrite(out_path, cropped)
            count += 1
            
    print(f"Successfully cropped {count} Front and Back panels to {output_dir}")

if __name__ == "__main__":
    main()
