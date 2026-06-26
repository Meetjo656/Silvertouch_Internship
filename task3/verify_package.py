import os
import yaml
from pathlib import Path

def main():
    package_dir = Path(r"d:\Silvertouch Internship\task3\training_package")
    
    # Verify dataset.yaml
    yaml_path = package_dir / "dataset" / "dataset.yaml"
    assert yaml_path.exists(), "dataset.yaml missing!"
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    print("YAML Content:", data)
    
    # Check paths
    for split in ["train", "val", "test"]:
        if split in data:
            # Paths are relative to dataset directory where yaml is
            split_dir = package_dir / "dataset" / data[split]
            assert split_dir.exists(), f"Path for {split} ({split_dir}) does not exist!"
            # Check if it has any images
            images = list(split_dir.iterdir())
            print(f"{split} has {len(images)} images.")
    
    # Check train.py references
    train_py_path = package_dir / "scripts" / "train.py"
    assert train_py_path.exists(), "train.py missing!"
    
    with open(train_py_path, "r", encoding="utf-8") as f:
        train_code = f.read()
        assert "dataset.yaml" in train_code, "train.py does not reference dataset.yaml!"
        assert "d:\\" not in train_code.lower() and "c:\\" not in train_code.lower(), "train.py has hardcoded Windows paths!"
        
    print("Paths and configurations verified successfully.")

if __name__ == "__main__":
    main()
