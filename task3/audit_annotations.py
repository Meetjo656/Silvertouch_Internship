"""Quick audit of existing YOLO annotation files."""
import os
from collections import Counter

CLASS_NAMES = {
    0: "Artwork", 1: "Not_Artwork", 2: "Front_Panel", 3: "Side_Panel",
    4: "Back_Panel", 5: "Extra_Areas", 6: "Bottom_Part", 7: "Top_Panel"
}

for subdir in ["Artwork", "Not_Artwork"]:
    ann_dir = rf"d:\Silvertouch Internship\task3\new (1)\annotations\{subdir}"
    txts = [f for f in os.listdir(ann_dir) if f.endswith(".txt")]

    empty = 0
    total_boxes = 0
    class_counts = Counter()
    bad_lines = 0
    bad_cls = Counter()
    confs = []
    out_of_range = 0

    for txt in txts:
        with open(os.path.join(ann_dir, txt)) as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            empty += 1
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                bad_lines += 1
                continue
            try:
                cls = int(parts[0])
                vals = [float(p) for p in parts[1:5]]
                conf = float(parts[5]) if len(parts) > 5 else None
            except (ValueError, IndexError):
                bad_lines += 1
                continue

            if cls not in CLASS_NAMES:
                bad_cls[cls] += 1
            class_counts[cls] += 1
            total_boxes += 1
            if conf is not None:
                confs.append(conf)

            for v in vals:
                if v < 0 or v > 1.001:
                    out_of_range += 1
                    break

    unk = "UNKNOWN"
    print(f"=== {subdir} ===")
    print(f"  Annotation files: {len(txts)}")
    print(f"  Empty files (0 detections): {empty}")
    print(f"  Total boxes: {total_boxes}")
    print(f"  Bad/malformed lines: {bad_lines}")
    print(f"  Out-of-range values: {out_of_range}")
    print(f"  Unknown class IDs: {dict(bad_cls)}")
    print(f"  Class distribution:")
    for cls_id in sorted(class_counts.keys()):
        name = CLASS_NAMES.get(cls_id, unk)
        print(f"    {cls_id} ({name}): {class_counts[cls_id]}")
    if confs:
        print(f"  Confidence range: {min(confs):.4f} - {max(confs):.4f}")
        print(f"  Mean confidence: {sum(confs)/len(confs):.4f}")
        low = sum(1 for c in confs if c < 0.3)
        print(f"  Detections below 0.3 conf: {low}")
        low5 = sum(1 for c in confs if c < 0.5)
        print(f"  Detections below 0.5 conf: {low5}")
    print()
