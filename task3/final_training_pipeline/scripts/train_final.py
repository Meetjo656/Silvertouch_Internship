from ultralytics import YOLO
import os
from pathlib import Path

def main():
    print("--- 5. Check train_final.py & 8. One thing I would change ---")
    # Setup paths
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "configs" / "dataset.yaml"
    
    # Initialize the YOLOv8m model
    model = YOLO("yolov8m.pt")
    
    # Start training
    model.train(
        data=str(config_path),
        epochs=200,      # Updated from 150 to 200 for better convergence on small datasets
        imgsz=640,
        batch=16,
        workers=8,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        weight_decay=0.0005,
        patience=40,     # Updated from 30 to 40
        device=0,
        project=str(base_dir / "outputs"),
        name="final_artwork_detector",
        pretrained=True,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        deterministic=True,
        seed=42,
        
        # Augmentation settings to preserve pharmaceutical artwork geometry
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.2, # 1.0 +/- 0.2 (0.8 - 1.2)
        flipud=0.0,
        fliplr=0.5,
        
        # Explicitly disabled augmentations per requirements
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        perspective=0.0,
        erasing=0.0
    )
    print("Training configuration initialized and ready.")

if __name__ == '__main__':
    main()
