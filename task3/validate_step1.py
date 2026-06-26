import json
import os
from pathlib import Path

def validate_dataset():
    dataset_dir = Path(r"d:\Silvertouch Internship\task3\dataset_split\assembled_cartons")
    images_dir = dataset_dir / "images"
    annotations_file = dataset_dir / "annotations_assembled.json"

    with open(annotations_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Assuming COCO format based on .json extension
    images = data.get('images', [])
    annotations = data.get('annotations', [])
    categories = data.get('categories', [])

    category_map = {c['id']: c['name'] for c in categories}
    
    total_images = len(images)
    total_annotations = len(annotations)

    image_ids = [img['id'] for img in images]
    image_filenames = {img['id']: img['file_name'] for img in images}
    
    # Check duplicate images
    duplicate_images = len(image_ids) - len(set(image_ids))
    
    # Check duplicate annotations
    annotation_ids = [ann['id'] for ann in annotations]
    duplicate_annotations = len(annotation_ids) - len(set(annotation_ids))

    missing_images_in_dir = []
    for img in images:
        if not (images_dir / img['file_name']).exists():
            missing_images_in_dir.append(img['file_name'])

    invalid_boxes = 0
    empty_annotations = 0 # In COCO this might mean images with no annotations
    
    img_has_annotations = {img['id']: 0 for img in images}
    
    per_class_anno_count = {c['name']: 0 for c in categories}
    per_class_img_count = {c['name']: set() for c in categories}

    for ann in annotations:
        img_id = ann['image_id']
        if img_id in img_has_annotations:
            img_has_annotations[img_id] += 1
        
        # Check bbox
        bbox = ann.get('bbox', [])
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            invalid_boxes += 1

        cat_id = ann['category_id']
        cat_name = category_map.get(cat_id, "Unknown")
        if cat_name in per_class_anno_count:
            per_class_anno_count[cat_name] += 1
            per_class_img_count[cat_name].add(img_id)

    empty_annotations = sum(1 for v in img_has_annotations.values() if v == 0)

    # Generate Report
    report_lines = []
    report_lines.append("# Assembled Dataset Statistics\n")
    report_lines.append(f"**Total images:** {total_images}")
    report_lines.append(f"**Total annotations:** {total_annotations}")
    
    report_lines.append("\n## Per-class Annotation Count")
    for cname, count in per_class_anno_count.items():
        report_lines.append(f"- {cname}: {count}")

    report_lines.append("\n## Per-class Image Count")
    for cname, img_set in per_class_img_count.items():
        report_lines.append(f"- {cname}: {len(img_set)}")

    report_lines.append("\n## Invalid Annotations")
    if invalid_boxes > 0:
        report_lines.append(f"- Invalid bounding boxes: {invalid_boxes}")
    else:
        report_lines.append("- None detected.")
        
    if duplicate_annotations > 0:
        report_lines.append(f"- Duplicate annotations: {duplicate_annotations}")

    if empty_annotations > 0:
        report_lines.append(f"- Images with empty annotations (no objects): {empty_annotations}")

    report_lines.append("\n## Missing Files")
    if missing_images_in_dir:
        report_lines.append(f"- {len(missing_images_in_dir)} missing images detected.")
    else:
        report_lines.append("- None detected.")

    report_content = "\n".join(report_lines)
    with open(r"d:\Silvertouch Internship\task3\assembled_dataset_statistics.md", "w", encoding='utf-8') as f:
        f.write(report_content)

    print("Statistics generated successfully.")

if __name__ == "__main__":
    validate_dataset()
