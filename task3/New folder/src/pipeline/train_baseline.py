from ultralytics import YOLO

def run_baseline():
    print("Starting Final YOLOv8m Baseline Experiment (imgsz=640)...")
    
    # Load medium model
    model = YOLO('yolov8m.pt')
    
    # Run full 75-epoch baseline
    results = model.train(
        data='d:/Silvertouch Internship/task3/yolo_detection_dataset/data.yaml',
        epochs=75,
        imgsz=640,
        batch=16,       # Resumed standard batch size for Medium model
        patience=20,
        device=0,
        workers=0,      # Required to prevent Windows multiprocessing crash
        plots=True,
        save=True,
        project='runs/detect',
        name='structural_yolov8m_640_baseline'
        # Note: Heavy augmentations (mosaic, mixup) are RE-ENABLED by default here for the real baseline
    )
    
    print("\nFinal Baseline Experiment completed successfully!")

if __name__ == '__main__':
    run_baseline()
