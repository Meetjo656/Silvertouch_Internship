import os
from PIL import Image

def process_images_in_place(target_dir, target_size=(640, 640)):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    if not os.path.exists(target_dir):
        print(f"Directory {target_dir} does not exist.")
        return

    image_paths = []
    for f in os.listdir(target_dir):
        if f.lower().endswith(valid_extensions):
            image_paths.append(os.path.join(target_dir, f))
            
    count = 0
    deleted = 0
    for path in image_paths:
        temp_path = path + ".tmp"
        try:
            with Image.open(path) as img:
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
                if img_resized.mode in ("RGBA", "P") and path.lower().endswith(('.jpg', '.jpeg')):
                    img_resized = img_resized.convert("RGB")
                
                img_format = img.format if img.format else 'JPEG'
                img_resized.save(temp_path, format=img_format)
                
            os.replace(temp_path, path)
            count += 1
            print(f"Resized {os.path.basename(path)}")
            
        except Exception as e:
            print(f"Error processing {os.path.basename(path)}, deleting. Reason: {e}")
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass
            try:
                os.remove(path)
                deleted += 1
            except Exception as e2:
                print(f"Failed to delete {path}: {e2}")

    print(f"Successfully resized {count} images.")
    print(f"Deleted {deleted} unresizable images.")

if __name__ == "__main__":
    target_dir = r"d:\Silvertouch Internship\task3\Scrapped\output\images"
    process_images_in_place(target_dir)
