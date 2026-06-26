import os
from pathlib import Path
from PIL import Image

IMAGE_DIRS = [
    r"d:\Silvertouch Internship\task3\new (1)\images\Artwork",
    r"d:\Silvertouch Internship\task3\new (1)\images\Not_Artwork",
]

def resize_images(target_size=(640, 640)):
    total_processed = 0
    total_deleted = 0
    
    for img_dir in IMAGE_DIRS:
        print(f"Processing directory: {img_dir}")
        for filename in os.listdir(img_dir):
            filepath = os.path.join(img_dir, filename)
            
            # Skip directories
            if not os.path.isfile(filepath):
                continue
                
            try:
                with Image.open(filepath) as img:
                    # Convert to RGB if it's not (e.g. RGBA or P) to save as JPEG safely
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        
                    # Resize ignoring aspect ratio as requested
                    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                    
                # Save over the original
                resized_img.save(filepath)
                total_processed += 1
                
            except Exception as e:
                print(f"Failed to open/resize {filename}: {e}. Deleting.")
                try:
                    os.remove(filepath)
                    total_deleted += 1
                except Exception as del_e:
                    print(f"Could not delete {filename}: {del_e}")
                    
    print(f"\nFinished resizing!")
    print(f"Total resized: {total_processed}")
    print(f"Total deleted: {total_deleted}")

if __name__ == '__main__':
    resize_images()
