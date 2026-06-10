import os
from collections import Counter

counter = Counter()
label_dir = r"d:\Silvertouch Internship\task3\yolo_detection_dataset\train\labels"

for file in os.listdir(label_dir):
    if not file.endswith(".txt"):
        continue

    with open(os.path.join(label_dir, file), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                cls = int(parts[0])
                counter[cls] += 1

print("Label Counts in Training Set:")
for cls_id, count in sorted(counter.items()):
    print(f"Class {cls_id}: {count}")