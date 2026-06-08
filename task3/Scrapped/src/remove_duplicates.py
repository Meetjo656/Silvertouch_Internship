import os
import hashlib

def remove_duplicate_images(target_dir):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    if not os.path.exists(target_dir):
        print(f"Directory {target_dir} does not exist.")
        return

    # To store hash -> file path
    hashes = {}
    duplicates = []
    
    image_paths = []
    for f in os.listdir(target_dir):
        if f.lower().endswith(valid_extensions):
            image_paths.append(os.path.join(target_dir, f))
            
    for path in image_paths:
        try:
            with open(path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
                
            if file_hash in hashes:
                duplicates.append(path)
                print(f"Duplicate found: {os.path.basename(path)} is same as {os.path.basename(hashes[file_hash])}")
            else:
                hashes[file_hash] = path
        except Exception as e:
            print(f"Error reading {path}: {e}")
            
    # Remove duplicates
    deleted = 0
    for dup in duplicates:
        try:
            os.remove(dup)
            deleted += 1
            print(f"Deleted {os.path.basename(dup)}")
        except Exception as e:
            print(f"Failed to delete {dup}: {e}")
            
    print(f"\nSuccessfully deleted {deleted} duplicate images out of {len(duplicates)} duplicates found.")
    print(f"Remaining unique images: {len(hashes)}")

if __name__ == "__main__":
    target_dir = r"d:\Silvertouch Internship\task3\Scrapped\output\images"
    remove_duplicate_images(target_dir)
