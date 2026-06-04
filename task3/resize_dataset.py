import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def resize_dataset(target_dirs, max_size=1000):
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    total_files_scanned = 0
    images_processed = 0
    images_resized = 0
    images_skipped = 0
    errors_encountered = 0
    
    # First, collect all valid image paths
    image_paths = []
    for d in target_dirs:
        dir_path = Path(d)
        if not dir_path.exists():
            print(f"Warning: Directory does not exist: {d}")
            continue
            
        for root, _, files in os.walk(dir_path):
            for file in files:
                total_files_scanned += 1
                if Path(file).suffix.lower() in valid_extensions:
                    image_paths.append(Path(root) / file)
                else:
                    images_skipped += 1

    print(f"Found {len(image_paths)} images to process.")
    
    # Process images with progress bar
    for img_path in tqdm(image_paths, desc="Resizing Images"):
        images_processed += 1
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                
                if max(width, height) <= max_size:
                    images_skipped += 1
                    continue
                    
                scale = max_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                # Use LANCZOS for high-quality downsampling
                resized_img = img.resize(
                    (new_width, new_height),
                    Image.Resampling.LANCZOS
                )
                
                # Save to a temporary variable to prevent overwriting issues on Windows while open
                # But PIL closes the file once out of context or loaded, so we just save
                # We need to preserve EXIF if possible, but standard is fine
                # Overwriting in place:
                # Need to convert to RGB if it's RGBA and saving as JPEG
                if resized_img.mode in ("RGBA", "P") and img_path.suffix.lower() in ('.jpg', '.jpeg'):
                    resized_img = resized_img.convert("RGB")
                    
                resized_img.save(img_path)
                
                # Log the change
                tqdm.write(f"Original: {img_path.name}")
                tqdm.write(f"Old Size: {width}x{height}")
                tqdm.write(f"New Size: {new_width}x{new_height}\n")
                
                images_resized += 1
                
        except Exception as e:
            tqdm.write(f"Error processing {img_path.name}: {e}")
            errors_encountered += 1
            images_skipped += 1

    # Print Summary
    print("\n" + "="*40)
    print("RESIZE SUMMARY")
    print("="*40)
    print(f"Total files scanned : {total_files_scanned}")
    print(f"Images processed    : {images_processed}")
    print(f"Images resized      : {images_resized}")
    print(f"Images skipped      : {images_skipped}")
    print(f"Errors encountered  : {errors_encountered}")
    print("="*40)

if __name__ == "__main__":
    directories_to_process = [
        r"D:\Silvertouch Internship\task3\Scrapped\output\images",
        r"D:\Silvertouch Internship\task3\train",
        r"D:\Silvertouch Internship\task3\test"
    ]
    
    resize_dataset(directories_to_process, max_size=1000)
