import json

JSON_PATH = r"d:\Silvertouch Internship\task3\dataset_split\assembled_cartons\annotations_assembled.json"

def main():
    print(f"Loading {JSON_PATH}...")
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
        
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    
    class_counts = {cat_id: 0 for cat_id in categories.keys()}
    
    for ann in data["annotations"]:
        class_counts[ann["category_id"]] += 1
        
    print("\n--- Class Balance on Assembled Cartons Only ---")
    for cat_id, count in class_counts.items():
        if cat_id in categories:
            print(f"{categories[cat_id]:<15}: {count}")

if __name__ == '__main__':
    main()
