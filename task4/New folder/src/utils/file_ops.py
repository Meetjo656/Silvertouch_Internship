import json
import os
from datetime import datetime
from pathlib import Path

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_json_artifact(data, output_dir, base_filename):
    """
    Saves a JSON artifact with a timestamp appended to the filename.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = get_timestamp()
    filename = f"{base_filename}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Artifact saved: {filepath}")
    return filepath

def save_markdown_artifact(content, output_dir, base_filename):
    """
    Saves a Markdown artifact with a timestamp appended to the filename.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = get_timestamp()
    filename = f"{base_filename}_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Artifact saved: {filepath}")
    return filepath

def load_json(filepath):
    """Utility to load a json file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
