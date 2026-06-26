import json
import random

JSON_PATH = r"d:\Silvertouch Internship\task3\new (1)\annotations\annotations_v3.json"

def main():
    print(f"Loading {JSON_PATH}...")
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
        
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    
    structural_classes = {3, 4, 5, 7, 8} # Front, Side, Back, Bottom, Top
    extra_areas_class = 6
    
    class_counts = {cat_id: 0 for cat_id in categories.keys()}
    annotations_by_img = {img["id"]: [] for img in data["images"]}
    img_by_id = {img["id"]: img["file_name"] for img in data["images"]}
    
    for ann in data["annotations"]:
        class_counts[ann["category_id"]] += 1
        annotations_by_img[ann["image_id"]].append(ann)
        
    print("\n--- Annotation Counts per Class ---")
    for cat_id, count in class_counts.items():
        if cat_id in categories:
            print(f"{categories[cat_id]:<15}: {count}")
        
    images_missing_front = []
    images_missing_back = []
    images_no_structural = []
    images_only_extra_areas = []
    cluttered_images = []
    
    for img_id, anns in annotations_by_img.items():
        classes_present = set(a["category_id"] for a in anns)
        
        if 3 not in classes_present:
            images_missing_front.append(img_id)
            
        if 5 not in classes_present:
            images_missing_back.append(img_id)
            
        if len(classes_present.intersection(structural_classes)) == 0:
            images_no_structural.append(img_id)
            
        if extra_areas_class in classes_present and len(classes_present.intersection(structural_classes)) == 0:
            images_only_extra_areas.append(img_id)
            
        if len(anns) >= 15:
            cluttered_images.append(img_id)

    total_images = len(data["images"])
    print(f"\n--- Image Statistics (Total images: {total_images}) ---")
    print(f"Images with missing Front        : {len(images_missing_front)} ({len(images_missing_front)/total_images*100:.2f}%)")
    print(f"Images with missing Back         : {len(images_missing_back)} ({len(images_missing_back)/total_images*100:.2f}%)")
    print(f"Images with no structural labels : {len(images_no_structural)} ({len(images_no_structural)/total_images*100:.2f}%)")
    print(f"Images with only Extra_Areas     : {len(images_only_extra_areas)} ({len(images_only_extra_areas)/total_images*100:.2f}%)")
    print(f"Images with >= 15 annotations (cluttered): {len(cluttered_images)} ({len(cluttered_images)/total_images*100:.2f}%)")

    # Step 2: Generate sample lists
    print("\n--- Step 2: Generating Sample Lists for Review ---")
    
    all_img_ids = list(img_by_id.keys())
    
    # 50 random images
    random_sample = random.sample(all_img_ids, min(50, len(all_img_ids)))
    
    # 50 images with suspected missing labels (missing front or back)
    suspected_missing = list(set(images_missing_front + images_missing_back) - set(images_no_structural))
    missing_sample = random.sample(suspected_missing, min(50, len(suspected_missing)))
    
    # 50 cluttered images
    clutter_sample = random.sample(cluttered_images, min(50, len(cluttered_images)))
    
    with open(r"d:\Silvertouch Internship\task3\sample_review_list.txt", "w") as f:
        f.write("--- 50 Random Images ---\n")
        for img_id in random_sample:
            f.write(f"{img_by_id[img_id]}\n")
            
        f.write("\n--- 50 Images with Suspected Missing Labels (Front or Back) ---\n")
        for img_id in missing_sample:
            f.write(f"{img_by_id[img_id]}\n")
            
        f.write("\n--- 50 Cluttered Images (>= 15 annotations) ---\n")
        for img_id in clutter_sample:
            f.write(f"{img_by_id[img_id]}\n")
            
    print("\nSample lists written to sample_review_list.txt")

if __name__ == '__main__':
    main()
