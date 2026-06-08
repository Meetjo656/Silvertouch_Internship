import sys
from pathlib import Path

# Add src to python path to allow absolute imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data.coco_to_yolo import convert_coco_to_yolo
from src.data.augment import augment_dataset
from src.data.create_yaml import create_yaml
from src.tools.validate_yolo_dataset import validate_dataset

def main():
    config_path = "config/yolo_dataset.yaml"
    
    print("Starting YOLO Detection Dataset Preparation Pipeline...")
    
    print("\n--- Step 1: Converting COCO to YOLO (Bounding Boxes & Preserving Splits) ---")
    convert_coco_to_yolo(config_path)
    
    print("\n--- Step 2: Augmenting Train Split (BBox Aware) ---")
    augment_dataset(config_path)
    
    print("\n--- Step 3: Generating data.yaml ---")
    create_yaml(config_path)
    
    print("\n--- Step 4: Validating Dataset & Generating Reports ---")
    validate_dataset(config_path)
    
    print("\nPipeline completed successfully! Check the generated markdown reports.")

if __name__ == "__main__":
    main()
