# Final YOLOv8 Training Pipeline

This pipeline is dedicated to the **final production training** for the pharmaceutical artwork detection model.

## Dataset Structure
This pipeline relies solely on the manually curated dataset. The dataset is derived from `New_Training/` and mapped strictly to two classes:
- `0`: Artwork
- `1`: not_artwork

All other classes have been intentionally excluded to ensure a high-quality production model for artwork detection.

## Conversion Process
1. **COCO to YOLO Conversion**: The script `scripts/coco_to_yolo.py` extracts the COCO JSON annotations. It ignores extraneous classes and produces corresponding YOLO format text labels alongside the images in a temporary directory.
2. **Dataset Splitting**: The script `scripts/split_dataset.py` divides the transformed dataset into Train (80%), Validation (10%), and Test (10%) splits with a fixed random seed (`42`) to ensure reproducibility.

To prepare the dataset for training, run the following sequentially:
```bash
python scripts/coco_to_yolo.py
python scripts/split_dataset.py
```
This will automatically generate the images and labels within the `dataset/` directory.

## Train/Validation/Test Split
- **Train**: 80%
- **Validation**: 10%
- **Test**: 10%
- **Seed**: 42

## Training Command
To start the model training procedure, execute:
```bash
python scripts/train_final.py
```
*Note: Ensure the dataset has been generated first and that you are executing this when the server GPU is ready. The pipeline is pre-configured and ready to run.*

## Expected Output Locations
Once training is initiated, the outputs (logs, weights, plots) will be stored in:
`outputs/final_artwork_detector/`

## Best Model Location
When the training completes successfully, the optimal model weights will be located at:
`outputs/final_artwork_detector/weights/best.pt`
