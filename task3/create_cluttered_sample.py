import os
import json
import shutil

JSON_PATH = r"d:\Silvertouch Internship\task3\new (1)\annotations\annotations_v3.json"
TXT_PATH = r"d:\Silvertouch Internship\task3\sample_review_list.txt"
OUTPUT_DIR = r"d:\Silvertouch Internship\task3\cluttered_sample"
IMG_BASE_DIRS = {
    "Artwork": r"d:\Silvertouch Internship\task3\new (1)\images\Artwork",
    "Not_Artwork": r"d:\Silvertouch Internship\task3\new (1)\images\Not_Artwork"
}

def resolve_path(filename):
    for subdir, path in IMG_BASE_DIRS.items():
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            return full_path
    return None

def main():
    # 1. Parse sample_review_list.txt to get the 50 cluttered images
    target_filenames = set()
    with open(TXT_PATH, "r") as f:
        lines = f.read().splitlines()
        
    in_section = False
    for line in lines:
        line = line.strip()
        if "--- 50 Cluttered Images" in line:
            in_section = True
            continue
        elif line.startswith("--- "):
            if in_section:
                break # Ended section
                
        if in_section and line and not line.startswith("---"):
            target_filenames.add(line)
            
    print(f"Found {len(target_filenames)} target filenames in the text file.")
    
    # 2. Make output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 3. Load annotations
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
        
    # 4. Filter images and annotations
    new_images = []
    target_image_ids = set()
    
    # Copy images and build new_images list
    for img in data["images"]:
        if img["file_name"] in target_filenames:
            new_images.append(img)
            target_image_ids.add(img["id"])
            
            # Copy the file
            src_path = resolve_path(img["file_name"])
            if src_path:
                dst_path = os.path.join(OUTPUT_DIR, img["file_name"])
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
            else:
                print(f"Warning: Could not find image file for {img['file_name']}")
                
    new_annotations = []
    for ann in data["annotations"]:
        if ann["image_id"] in target_image_ids:
            new_annotations.append(ann)
            
    # 5. Create new JSON
    new_data = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": data["categories"]
    }
    if "info" in data:
        new_data["info"] = data["info"]
    if "licenses" in data:
        new_data["licenses"] = data["licenses"]
        
    out_json = os.path.join(OUTPUT_DIR, "annotations.json")
    with open(out_json, "w") as f:
        json.dump(new_data, f, indent=2)
        
    print(f"Successfully copied images and created {out_json}")

if __name__ == '__main__':
    main()
