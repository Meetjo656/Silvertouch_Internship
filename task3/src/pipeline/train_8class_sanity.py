from ultralytics import YOLO
import os

def main():
    print("Starting 8-Class Sanity Check Training...")
    
    # 1. Load the model
    model_path = r"d:\Silvertouch Internship\task3\yolo26n.pt"
    model = YOLO(model_path)
    
    # 2. Define dataset path
    data_yaml = r"d:\Silvertouch Internship\task3\yolo_detection_dataset\data.yaml"
    
    # 3. Train
    results = model.train(
        data=data_yaml,
        epochs=5,
        imgsz=512,
        batch=16,
        project=r"d:\Silvertouch Internship\task3\trained_models",
        name="8class_sanity_check",
        workers=0,  # CRITICAL for Windows to avoid multiprocessing crash (WinError 1455)
        plots=True
    )
    
    print("Sanity check training completed successfully!")

if __name__ == '__main__':
    main()
