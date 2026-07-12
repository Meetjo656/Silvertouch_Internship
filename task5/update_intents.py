import json
from pathlib import Path
import glob

base_dir = Path(r"d:\Silvertouch Internship\task5")
files_to_update = [
    base_dir / "training_data.json"
]

for p in (base_dir / "training_data").glob("*.json"):
    files_to_update.append(p)

new_examples = [
    "Check my MOU application status ID = ",
    "Check my MOU application status ID: - ",
    "Check my MOU application status",
    "MOU application status",
    "where is my MOU application",
    "MOU application ID =",
    "MOU application ID: -",
    "track MOU application"
]

for file_path in files_to_update:
    if not file_path.exists():
        continue
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        updated = False
        for sub_intent in data.get("subIntents", []):
            if sub_intent.get("name") == "application_status":
                examples = sub_intent.setdefault("examples", [])
                for ne in new_examples:
                    if ne not in examples:
                        examples.append(ne)
                        updated = True
        
        if updated:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"Updated {file_path.name}")
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
