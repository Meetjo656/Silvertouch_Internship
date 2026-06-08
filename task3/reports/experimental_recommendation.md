# Experimental Recommendation

Based on dataset statistics, the following sequential experiments are recommended. 
Do not assume YOLOv8l wins. Baseline models often perform best on smaller datasets.

### Experiment 1: Baseline
- **Model**: YOLOv8m
- **Image Size**: `imgsz=640`
- **Purpose**: Establishes baseline precision/recall without high risk of overfitting.

### Experiment 2: Image Size Scaling Test
- **Model**: YOLOv8m
- **Image Size**: `imgsz=800`
- **Purpose**: Evaluates impact of higher resolution, critical for datasets with high object density and small objects.

### Experiment 3: Model Capacity Scaling Test
- **Model**: YOLOv8l
- **Image Size**: `imgsz=640`
- **Purpose**: Evaluates whether additional parameters aid in resolving the 20 classes, or simply induce overfitting.

### Comparison Metrics
Compare the following across all 3 experiments to declare a winner:
- Precision
- Recall
- mAP50
- mAP50-95
- Training Time
- Inference Time
