# Baseline Training Recommendation

## Experiment 1: The Fast Baseline
```python
model = YOLO("yolov8m.pt")
model.train(data="yolo_detection_dataset/data.yaml",
            imgsz=640,
            epochs=250,
            batch=16,
            patience=50,
            save_best=True,
            plots=True)
```
**Justification**:
- `YOLOv8m`: Provides the best balance of speed and accuracy for a medium-complexity structural task.
- `imgsz=640`: The default standard. Keeps training fast while allowing the network to learn global structural shapes.
- `epochs=250`: Gives the model plenty of runway to converge.
- `patience=50`: Critical parameter. Since overfitting is a known risk (due to 511 original images), this ensures training stops early if validation mAP plateaus for 50 epochs, saving the best weights.
- `batch=16`: Standard stable batch size for a 640px image resolution.

## Experiment 2: The High-Resolution Enhancer
```python
model = YOLO("yolov8m.pt")
model.train(..., imgsz=800)
```
**Expected Benefits**:
Structural packaging boundaries (especially `Top_Panel` and `Bottom_Part` flaps) can be extremely thin or ambiguous at standard resolutions. Bumping `imgsz` to 800 allows the convolutional layers to preserve finer edge gradients, potentially boosting recall on the minority panel classes at the cost of slower inference.

## Experiment 3: The High-Capacity Scaler
```python
model = YOLO("yolov8l.pt")
model.train(..., imgsz=800)
```
**Expected Benefits**:
The deeper, wider architecture of YOLOv8l combined with high resolution allows the model to learn incredibly nuanced feature representations, which might better distinguish visually similar panels (e.g., `Front_Panel` vs `Back_Panel` based on complex contextual clues).

**Expected Risks**:
Massive risk of overfitting. The model's capacity heavily outclasses the diversity of the 511 original images. Early stopping will likely trigger much faster, and validation loss might diverge while training loss approaches zero.
