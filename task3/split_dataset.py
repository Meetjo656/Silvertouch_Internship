import os
import json
import shutil

JSON_PATH = r"d:\Silvertouch Internship\task3\new (1)\annotations\annotations_v3.json"
IMG_BASE_DIRS = {
    "Artwork": r"d:\Silvertouch Internship\task3\new (1)\images\Artwork",
    "Not_Artwork": r"d:\Silvertouch Internship\task3\new (1)\images\Not_Artwork"
}
OUTPUT_BASE_DIR = r"d:\Silvertouch Internship\task3\dataset_split"
ASSEMBLED_DIR = os.path.join(OUTPUT_BASE_DIR, "assembled_cartons")
DIELINES_DIR = os.path.join(OUTPUT_BASE_DIR, "dielines")

def resolve_path(filename):
    for subdir, path in IMG_BASE_DIRS.items():
        if " - Copy" in filename: continue
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            return full_path
    return None

def main():
    os.makedirs(os.path.join(ASSEMBLED_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(DIELINES_DIR, "images"), exist_ok=True)
    
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
        
    img_by_id = {img["id"]: img for img in data["images"]}
    anns_by_img = {img["id"]: [] for img in data["images"]}
    for ann in data["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)
        
    assembled_images = []
    assembled_anns = []
    
    dielines_images = []
    dielines_anns = []
    
    missing_files = 0
    
    print("Partitioning dataset. Please wait, copying images...")
    
    for img_id, img_info in img_by_id.items():
        anns = anns_by_img[img_id]
        
        src_path = resolve_path(img_info["file_name"])
        if not src_path:
            missing_files += 1
            continue
            
        if len(anns) >= 12:
            # Dieline
            dielines_images.append(img_info)
            dielines_anns.extend(anns)
            dst_path = os.path.join(DIELINES_DIR, "images", img_info["file_name"])
        else:
            # Assembled
            assembled_images.append(img_info)
            assembled_anns.extend(anns)
            dst_path = os.path.join(ASSEMBLED_DIR, "images", img_info["file_name"])
            
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
            
    # Write JSONs
    assembled_data = {
        "images": assembled_images,
        "annotations": assembled_anns,
        "categories": data["categories"]
    }
    if "info" in data: assembled_data["info"] = data["info"]
    if "licenses" in data: assembled_data["licenses"] = data["licenses"]
    
    dielines_data = {
        "images": dielines_images,
        "annotations": dielines_anns,
        "categories": data["categories"]
    }
    if "info" in data: dielines_data["info"] = data["info"]
    if "licenses" in data: dielines_data["licenses"] = data["licenses"]
    
    with open(os.path.join(ASSEMBLED_DIR, "annotations_assembled.json"), "w") as f:
        json.dump(assembled_data, f, indent=2)
        
    with open(os.path.join(DIELINES_DIR, "annotations_dielines.json"), "w") as f:
        json.dump(dielines_data, f, indent=2)
        
    print(f"Partitioning complete!")
    print(f"Assembled Cartons: {len(assembled_images)} images")
    print(f"Dielines (>=12 annotations): {len(dielines_images)} images")
    if missing_files > 0:
        print(f"Warning: {missing_files} files could not be found and were skipped.")

if __name__ == '__main__':
    main()
