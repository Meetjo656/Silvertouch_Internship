import os
import shutil

def flatten_dataset():
    base_dir = r"d:\Silvertouch Internship\task3\final_training_v2"
    
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        img_dir = os.path.join(split_dir, "images")
        if os.path.exists(img_dir):
            # Move all images out
            for f in os.listdir(img_dir):
                shutil.move(os.path.join(img_dir, f), os.path.join(split_dir, f))
            os.rmdir(img_dir)
            
        ann_file = os.path.join(split_dir, "annotations.json")
        if os.path.exists(ann_file):
            os.rename(ann_file, os.path.join(split_dir, "_annotations.coco.json"))
            
    print("Dataset flattened successfully to match Roboflow .coco reference.")

if __name__ == "__main__":
    flatten_dataset()
