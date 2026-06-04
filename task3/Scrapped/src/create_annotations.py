import os
import json
import datetime
from PIL import Image
from pathlib import Path

def create_annotations():
    base_dir = Path(r"d:\Silvertouch Internship\task3")
    scrapped_images_dir = base_dir / "Scrapped" / "output" / "images"
    train_json_path = base_dir / "train" / "_annotations.coco.json"
    output_json_path = scrapped_images_dir / "_annotations.coco.json"
    
    # Read categories from train dataset
    categories = []
    if train_json_path.exists():
        with open(train_json_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            categories = train_data.get("categories", [])
    
    info = {
        "year": str(datetime.datetime.now().year),
        "version": "1",
        "description": "Scraped DailyMed Artwork Dataset",
        "contributor": "",
        "url": "",
        "date_created": datetime.datetime.now().isoformat()
    }
    
    licenses = [
        {
            "id": 1,
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "name": "CC BY 4.0"
        }
    ]
    
    images = []
    image_id = 0
    
    if scrapped_images_dir.exists():
        for filename in os.listdir(scrapped_images_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = scrapped_images_dir / filename
                try:
                    with Image.open(filepath) as img:
                        width, height = img.size
                        
                    images.append({
                        "id": image_id,
                        "license": 1,
                        "file_name": filename,
                        "height": height,
                        "width": width,
                        "date_captured": datetime.datetime.now().isoformat(),
                        "extra": {
                            "name": filename
                        }
                    })
                    image_id += 1
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    
    coco_format = {
        "info": info,
        "licenses": licenses,
        "categories": categories,
        "images": images,
        "annotations": []
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_format, f, indent=4)
        
    print(f"Successfully created {output_json_path} with {len(images)} images.")

if __name__ == "__main__":
    create_annotations()
