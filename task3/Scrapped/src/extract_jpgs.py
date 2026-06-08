import os
import zipfile
import shutil

source_dirs = [
    r"D:\Silvertouch Internship\task3\Scrapped\output\New folder\otc",
    r"D:\Silvertouch Internship\task3\Scrapped\output\New folder\other"
]
target_dir = r"D:\Silvertouch Internship\task3\Scrapped\images"

# Ensure target directory exists
os.makedirs(target_dir, exist_ok=True)

extracted_count = 0

def get_unique_filename(target_path):
    base, ext = os.path.splitext(target_path)
    counter = 1
    new_path = target_path
    while os.path.exists(new_path):
        new_path = f"{base}_{counter}{ext}"
        counter += 1
    return new_path

for src_dir in source_dirs:
    if not os.path.exists(src_dir):
        print(f"Warning: Source directory not found - {src_dir}")
        continue
        
    print(f"\nScanning directory: {src_dir}")
    for filename in os.listdir(src_dir):
        if filename.lower().endswith(".zip"):
            zip_path = os.path.join(src_dir, filename)
            print(f"  Processing zip: {filename}")
            
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        if member.lower().endswith(".jpg") or member.lower().endswith(".jpeg"):
                            # Get the base filename without the zip's internal directory structure
                            member_filename = os.path.basename(member)
                            if not member_filename: # skip directory entries
                                continue
                                
                            target_filepath = os.path.join(target_dir, member_filename)
                            target_filepath = get_unique_filename(target_filepath)
                            
                            with zip_ref.open(member) as source_file:
                                with open(target_filepath, "wb") as target_file:
                                    shutil.copyfileobj(source_file, target_file)
                                    
                            extracted_count += 1
            except zipfile.BadZipFile:
                print(f"  Error: Bad or corrupted zip file - {filename}")
            except Exception as e:
                print(f"  Error processing {filename}: {e}")

print(f"\nExtraction complete! Extracted {extracted_count} JPG files to {target_dir}")
