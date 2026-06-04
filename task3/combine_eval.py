import json
import os
import shutil
from pathlib import Path

def main():
    base_dir = Path(r"d:\Silvertouch Internship\task3")
    test_dir = base_dir / "test"
    valid_dir = base_dir / "valid"
    out_dir = base_dir / "eval_combined"
    
    out_dir.mkdir(exist_ok=True)
    
    test_json_path = test_dir / "_annotations.coco.json"
    valid_json_path = valid_dir / "_annotations.coco.json"
    out_json_path = out_dir / "_annotations_mapped.json"
    
    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        
    with open(valid_json_path, 'r', encoding='utf-8') as f:
        valid_data = json.load(f)
        
    new_categories = [
        {"id": 1, "name": "Artwork", "supercategory": ""},
        {"id": 2, "name": "not_artwork", "supercategory": ""},
        {"id": 3, "name": "Front_Panel", "supercategory": ""},
        {"id": 4, "name": "Side_Panel", "supercategory": ""},
        {"id": 5, "name": "Back_Panel", "supercategory": ""},
        {"id": 6, "name": "Extra_Areas", "supercategory": ""},
        {"id": 7, "name": "Bottom_Part", "supercategory": ""},
        {"id": 8, "name": "Top_Panel", "supercategory": ""}
    ]

    category_mapping = {
        7: 3,
        15: 4,
        2: 5,
        3: 7,
        19: 8,
        6: 6
    }
    
    combined_images = []
    combined_annotations = []
    
    new_image_id = 1
    new_ann_id = 1
    
    # Process test dataset
    for img in test_data.get("images", []):
        old_id = img["id"]
        img["id"] = new_image_id
        combined_images.append(img)
        
        # Copy image file
        src_img = test_dir / img["file_name"]
        dst_img = out_dir / img["file_name"]
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
            
        # Find annotations for this image
        for ann in test_data.get("annotations", []):
            if ann["image_id"] == old_id:
                old_cat_id = ann.get("category_id")
                if old_cat_id in category_mapping:
                    ann["id"] = new_ann_id
                    ann["image_id"] = new_image_id
                    ann["category_id"] = category_mapping[old_cat_id]
                    combined_annotations.append(ann)
                    new_ann_id += 1
                    
        new_image_id += 1
        
    # Process valid dataset
    for img in valid_data.get("images", []):
        old_id = img["id"]
        img["id"] = new_image_id
        
        # Handle potential filename collisions (rare in roboflow, but possible)
        # If it happens, we rename the file.
        src_img = valid_dir / img["file_name"]
        dst_img = out_dir / img["file_name"]
        
        if dst_img.exists():
            new_name = f"valid_{img['file_name']}"
            img["file_name"] = new_name
            dst_img = out_dir / new_name
            
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
            
        combined_images.append(img)
        
        for ann in valid_data.get("annotations", []):
            if ann["image_id"] == old_id:
                old_cat_id = ann.get("category_id")
                if old_cat_id in category_mapping:
                    ann["id"] = new_ann_id
                    ann["image_id"] = new_image_id
                    ann["category_id"] = category_mapping[old_cat_id]
                    combined_annotations.append(ann)
                    new_ann_id += 1
                    
        new_image_id += 1

    combined_data = {
        "info": test_data.get("info", {}),
        "licenses": test_data.get("licenses", []),
        "categories": new_categories,
        "images": combined_images,
        "annotations": combined_annotations
    }
    
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=4)
        
    print(f"Total images combined: {len(combined_images)}")
    print(f"Total mapped annotations: {len(combined_annotations)}")
    print(f"Saved to: {out_json_path}")

if __name__ == "__main__":
    main()
