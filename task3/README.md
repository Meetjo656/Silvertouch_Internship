# Assembled Carton Detection Training Package

## Package Contents
- `dataset/`: Contains images and YOLO format annotations split into train, valid, and test sets, along with `dataset.yaml`.
- `scripts/`: Contains the training script (`train.py`) and dependencies list (`requirements.txt`).
- `documentation/`: Contains dataset statistics and this README.
- `outputs/`: Output directory where trained weights and logs will be stored.

## Expected Directory Structure
```
training_package/
├── dataset/
│   ├── train/ (images/ and labels/)
│   ├── valid/ (images/ and labels/)
│   ├── test/ (images/ and labels/)
│   └── dataset.yaml
├── scripts/
│   ├── train.py
│   └── requirements.txt
├── documentation/
│   ├── assembled_dataset_statistics.md
│   └── README.md
└── outputs/
```

## Environment Requirements
- Python 3.8+
- GPU recommended (NVIDIA)

## Setup & Training
1. Copy this entire `training_package/` directory to the GPU server.
2. Install dependencies:
   ```bash
   pip install -r scripts/requirements.txt
   ```
3. Launch training:
   ```bash
   python scripts/train.py
   ```
