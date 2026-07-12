# RF-DETR v2 Production Training Pipeline

This pipeline builds the 2nd-generation detection system using Roboflow's **RF-DETR Small** architecture. It is strictly isolated from the previous YOLO pipeline.

## Features
- **Official RF-DETR:** Utilizes the official `rfdetr` python package.
- **Strict Size Enforcement:** 640x640 preprocessing enforced globally.
- **Configurable Dataset:** COCO JSON pathways configured via `configs/dataset_config.yaml`. No duplicated datasets.
- **Experiment Tracking:** Captures detailed metadata (`experiment.yaml`) and data manifestations per run.
- **Production Ready:** Includes `pipeline.py` for direct OCR/Rx integrations.

## Commands

### 1. Verification & Insights
```bash
python scripts/verify_dataset.py
python scripts/dataset_statistics.py
python scripts/create_dataset_manifest.py
```

### 2. Training
Adjust hyperparameters in `configs/train.yaml`.
```bash
python scripts/train.py
```

### 3. Evaluation & Diagnostics
```bash
python scripts/evaluate.py --weights path/to/weights.pt
python scripts/evaluate_blind.py --weights path/to/weights.pt --source path/to/images/
python scripts/hard_example_mining.py --source path/to/images/ --preds outputs/blind_evaluation
```

### 4. Inference
Configured via `configs/inference.yaml`.
```bash
python scripts/predict.py --source path/to/images --weights path/to/weights.pt
```

### 5. Utilities
```bash
python scripts/visualize_predictions.py --image img.jpg --preds preds.json
python scripts/benchmark.py --weights path/to/weights.pt --source path/to/images
python scripts/export.py --weights path/to/weights.pt --format onnx
```
