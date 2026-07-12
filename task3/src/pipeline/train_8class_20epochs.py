from ultralytics import YOLO

def main():
    print("Starting 15-Epoch 8-Class Training...")
    
    # Load the base model
    model_path = r"d:\Silvertouch Internship\task3\yolo26n.pt"
    model = YOLO(model_path)
    
    # Dataset
    data_yaml = r"d:\Silvertouch Internship\task3\yolo_detection_dataset\data.yaml"
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=20,
        imgsz=512,
        batch=16,
        project=r"d:\Silvertouch Internship\task3\trained_models",
        name="structural_8class_20epochs",
        workers=0,  # CRITICAL for Windows
        plots=True
    )
    
    print("20-epoch training completed successfully!")

if __name__ == '__main__':
    main()
