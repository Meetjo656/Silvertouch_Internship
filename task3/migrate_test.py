import json

input_file = r'd:\Silvertouch Internship\task3\test\_annotations.coco.json'
output_file = r'd:\Silvertouch Internship\task3\test\_annotations_mapped.json'

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

new_categories = [
    {'id': 1, 'name': 'Artwork', 'supercategory': ''},
    {'id': 2, 'name': 'not_artwork', 'supercategory': ''},
    {'id': 3, 'name': 'Front_Panel', 'supercategory': ''},
    {'id': 4, 'name': 'Side_Panel', 'supercategory': ''},
    {'id': 5, 'name': 'Back_Panel', 'supercategory': ''},
    {'id': 6, 'name': 'Extra_Areas', 'supercategory': ''},
    {'id': 7, 'name': 'Bottom_Part', 'supercategory': ''},
    {'id': 8, 'name': 'Top_Panel', 'supercategory': ''}
]

category_mapping = {
    7: 3,
    15: 4,
    2: 5,
    3: 7,
    19: 8,
    6: 6
}

migrated_annotations = []
new_ann_id = 1

for ann in data.get('annotations', []):
    old_cat_id = ann.get('category_id')
    if old_cat_id in category_mapping:
        ann['id'] = new_ann_id
        ann['category_id'] = category_mapping[old_cat_id]
        migrated_annotations.append(ann)
        new_ann_id += 1

data['categories'] = new_categories
data['annotations'] = migrated_annotations

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)

print(f"Total annotations checked: {len(data.get('annotations'))}")
print(f"Total annotations kept and migrated: {len(migrated_annotations)}")
print(f"Saved to: {output_file}")
