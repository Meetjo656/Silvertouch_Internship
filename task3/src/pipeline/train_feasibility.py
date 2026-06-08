from ultralytics import YOLO

def run_fast_feasibility():
    print("Starting YOLOv8n Fast Feasibility Experiment...")
    
    # Use Nano model for rapid validation
    model = YOLO('yolov8n.pt')
    
    # Run heavily constrained 10-epoch validation
    results = model.train(
        data='d:/Silvertouch Internship/task3/yolo_detection_dataset/data.yaml',
        epochs=10,
        imgsz=512,      # Reduced resolution
        batch=8,        # Reduced VRAM pressure
        patience=10,
        device=0,
        workers=0,      # Required for Windows fallback
        plots=True,
        save=True,
        project='runs/detect',
        name='structural_yolov8n_512_feasibility',
        
        # Disable heavy bottleneck augmentations
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0
    )
    
    print("\nFast Feasibility Experiment completed successfully!")

if __name__ == '__main__':
    run_fast_feasibility()
