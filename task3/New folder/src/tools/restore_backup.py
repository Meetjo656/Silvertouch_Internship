import os
import shutil

base_dir = r"d:\Silvertouch Internship\task3"
dataset_dir = os.path.join(base_dir, "yolo_detection_dataset")
backup_dir = os.path.join(base_dir, "yolo_detection_dataset_backup")

print("Fixing file permissions...")
os.system(f'attrib -r "{dataset_dir}\\*.*" /s')
os.system(f'attrib -r "{backup_dir}\\*.*" /s')

if os.path.exists(dataset_dir):
    print("Removing corrupted dataset...")
    shutil.rmtree(dataset_dir)

print("Restoring from backup...")
shutil.copytree(backup_dir, dataset_dir)
print("Restore successful!")
