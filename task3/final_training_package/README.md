# 8-Class Final Production YOLOv8m Pipeline

This package contains the fully verified and merged 8-class dataset for structural artwork detection, along with the automated execution pipeline.

## Usage on GPU Server
1. Extract `final_training_package.zip` on the GPU server.
2. Monitor GPU and launch automatically:
   ```bash
   python scripts/gpu_waiter.py
   ```
3. Alternatively, launch training instantly:
   ```bash
   python scripts/start_training.py
   ```

*Note: The system automatically verifies the 8-class dataset ontology and integrity before any training executes.*

## Outputs
All checkpoints and plots will be placed automatically in `outputs/final_artwork_detector/`.
