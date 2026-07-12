import os
import shutil
from pathlib import Path

def main():
    base_dir = Path(r"d:\Silvertouch Internship\task3")
    package_dir = base_dir / "training_package"
    
    # 1. Create directory structure
    if package_dir.exists():
        shutil.rmtree(package_dir)
        
    (package_dir / "dataset").mkdir(parents=True)
    for split in ["train", "valid", "test"]:
        (package_dir / "dataset" / split / "images").mkdir(parents=True)
        (package_dir / "dataset" / split / "labels").mkdir(parents=True)
        
    (package_dir / "scripts" / "utils").mkdir(parents=True)
    (package_dir / "scripts" / "models").mkdir(parents=True)
    (package_dir / "scripts" / "configs").mkdir(parents=True)
    (package_dir / "documentation").mkdir(parents=True)
    (package_dir / "outputs").mkdir(parents=True)
    
    # 2. Copy dataset
    yolo_dir = base_dir / "dataset_split" / "yolo_format"
    
    splits_map = {"train": "train", "val": "valid", "test": "test"}
    for src_split, dst_split in splits_map.items():
        src_images = yolo_dir / "images" / src_split
        src_labels = yolo_dir / "labels" / src_split
        
        if src_images.exists():
            for f in src_images.iterdir():
                shutil.copy2(f, package_dir / "dataset" / dst_split / "images" / f.name)
        if src_labels.exists():
            for f in src_labels.iterdir():
                shutil.copy2(f, package_dir / "dataset" / dst_split / "labels" / f.name)
                
    # 3. Create dataset.yaml
    dataset_yaml_content = """train: train/images
val: valid/images
test: test/images

nc: 8
names:
  0: Artwork
  1: not_artwork
  2: Front_Panel
  3: Side_Panel
  4: Back_Panel
  5: Extra_Areas
  6: Bottom_Panel
  7: Top_Panel
"""
    with open(package_dir / "dataset" / "dataset.yaml", "w", encoding="utf-8") as f:
        f.write(dataset_yaml_content)
        
    # 4. Create train.py
    train_py_content = """import os
from ultralytics import YOLO

def main():
    # Set paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(script_dir)
    dataset_yaml = os.path.join(package_dir, "dataset", "dataset.yaml")
    output_dir = os.path.join(package_dir, "outputs")
    
    # Initialize YOLOv8 nano model
    model = YOLO("yolov8n.pt")
    
    # Train the model
    results = model.train(
        data=dataset_yaml,
        epochs=100,
        imgsz=640,
        batch=16,
        project=output_dir,
        name="structural_assembled",
        device="0"  # Assumes GPU training
    )
    
    print("Training complete. Results saved in:", output_dir)

if __name__ == "__main__":
    main()
"""
    with open(package_dir / "scripts" / "train.py", "w", encoding="utf-8") as f:
        f.write(train_py_content)
        
    # 5. Copy requirements.txt
    shutil.copy2(base_dir / "requirements.txt", package_dir / "scripts" / "requirements.txt")
    
    # 6. Copy documentation
    stats_file = base_dir / "assembled_dataset_statistics.md"
    if stats_file.exists():
        shutil.copy2(stats_file, package_dir / "documentation" / "assembled_dataset_statistics.md")
        
    # Generate README.md
    readme_content = """# Assembled Carton Detection Training Package

## Package Contents
- `dataset/`: Contains images and YOLO format annotations split into train, valid, and test sets, along with `dataset.yaml`.
- `scripts/`: Contains the training script (`train.py`) and dependencies list (`requirements.txt`).
- `documentation/`: Contains dataset statistics and this README.
- `outputs/`: Output directory where trained weights and logs will be stored.

## Expected Directory Structure
```
training_package/
├── dataset/
│   ├── train/ (images/ and labels/)
│   ├── valid/ (images/ and labels/)
│   ├── test/ (images/ and labels/)
│   └── dataset.yaml
├── scripts/
│   ├── train.py
│   └── requirements.txt
├── documentation/
│   ├── assembled_dataset_statistics.md
│   └── README.md
└── outputs/
```

## Environment Requirements
- Python 3.8+
- GPU recommended (NVIDIA)

## Setup & Training
1. Copy this entire `training_package/` directory to the GPU server.
2. Install dependencies:
   ```bash
   pip install -r scripts/requirements.txt
   ```
3. Launch training:
   ```bash
   python scripts/train.py
   ```
"""
    with open(package_dir / "documentation" / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
        
    print("Training package built successfully.")

if __name__ == "__main__":
    main()
