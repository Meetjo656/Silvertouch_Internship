from ultralytics import YOLO
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "dataset" / "dataset.yaml"
    outputs_dir = base_dir / "outputs"
    last_pt = outputs_dir / "final_artwork_detector" / "weights" / "last.pt"
    yolov8m_pt = base_dir / "weights" / "yolov8m.pt"

    if args.resume and last_pt.exists():
        print("Resuming training...")
        model = YOLO(str(last_pt))
        model.train(resume=True)
    else:
        print("Starting fresh training...")
        model = YOLO(str(yolov8m_pt))
        model.train(
            data=str(config_path),
            epochs=200,
            imgsz=640,
            batch=16,
            optimizer="AdamW",
            lr0=0.001,
            cos_lr=True,
            patience=40,
            workers=8,
            pretrained=True,
            seed=42,
            deterministic=True,
            project=str(outputs_dir),
            name="final_artwork_detector",
            exist_ok=True,
            fliplr=0.5,
            degrees=10.0,
            translate=0.1,
            scale=0.2,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            erasing=0.0,
            perspective=0.0
        )

if __name__ == '__main__':
    main()
