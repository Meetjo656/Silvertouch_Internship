from ultralytics import YOLO
import os
from pathlib import Path

def main():
    base_dir = Path.cwd()
    
    # Selected winning checkpoint
    model_weights = base_dir / "trained_models/structural_8class_50epochs/weights/best.pt"
    
    # Original ground truth dataset
    data_yaml = base_dir / "yolo_detection_dataset/data.yaml"
    
    print("Preparing to load model from:", model_weights)
    
    model = YOLO(str(model_weights))
    
    print("Starting A2 Fine-Tuning...")
    results = model.train(
        data=str(data_yaml),
        epochs=200,
        imgsz=800,
        patience=50,
        batch=-1,
        device=0,
        project=str(base_dir / "trained_models"),
        name="structural_8class_A2_200epochs_800",
        exist_ok=True,
        fliplr=0.0,
        flipud=0.0
    )
    
    print("Training completed successfully.")

if __name__ == "__main__":
    main()
