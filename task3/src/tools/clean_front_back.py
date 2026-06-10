import os
import json

def main():
    base_dir = r"d:\Silvertouch Internship\task3"
    json_path = os.path.join(base_dir, r"train\_annotations_mapped.json")
    front_back_dir = os.path.join(base_dir, "front&back")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Get the original file names (lowest ID for each extra.name)
    # This correctly identifies the unaugmented original image before Roboflow applied Flips
    orig_map = {}
    for img in data["images"]:
        if "extra" in img and "name" in img["extra"]:
            base_name = img["extra"]["name"]
            img_id = img["id"]
            if base_name not in orig_map or img_id < orig_map[base_name][0]:
                orig_map[base_name] = (img_id, img["file_name"])
                
    original_hashes = set(val[1] for val in orig_map.values())
    
    print(f"Identified {len(original_hashes)} pure original base images from JSON metadata.")
    
    # Process front&back directory
    deleted_count = 0
    kept_count = 0
    
    for filename in os.listdir(front_back_dir):
        filepath = os.path.join(front_back_dir, filename)
        if not os.path.isfile(filepath):
            continue
            
        # 1. Delete if it has _aug (our photometric augmentations and 90-deg rotations)
        if "_aug" in filename:
            os.remove(filepath)
            deleted_count += 1
            continue
            
        # 2. Check if it matches an original hash (filters out Roboflow horizontal/vertical flips)
        is_original = False
        for orig_hash in original_hashes:
            if orig_hash in filename:
                is_original = True
                break
                
        if not is_original:
            os.remove(filepath)
            deleted_count += 1
        else:
            kept_count += 1
            
    print(f"Cleaned front&back folder:")
    print(f" - Kept {kept_count} pure original images")
    print(f" - Deleted {deleted_count} augmented/flipped/mirrored images")

if __name__ == "__main__":
    main()
