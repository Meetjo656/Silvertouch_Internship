from ultralytics import YOLO

def run_experiment_1():
    print("Starting YOLOv8m Baseline Experiment (imgsz=640)...")
    
    # Load a medium YOLOv8 model
    model = YOLO('yolov8m.pt')
    
    # Run Baseline Experiment
    results = model.train(
        data='d:/Silvertouch Internship/task3/yolo_detection_dataset/data.yaml',
        epochs=250,
        imgsz=640,
        batch=16,
        patience=50,
        save=True,      # Saves best and last weights
        plots=True,     # Generates necessary evaluation PNGs
        cache=False,
        project='runs/detect',
        name='structural_yolov8m_640'
    )
    
    print("\nExperiment 1 completed successfully!")

if __name__ == '__main__':
    run_experiment_1()
