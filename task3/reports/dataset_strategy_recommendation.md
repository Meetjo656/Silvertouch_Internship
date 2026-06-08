# Dataset Strategy Recommendation

## Option A: Single YOLO Detector
Trains a single YOLO model to detect the 6 structural panels, plus uses `not_artwork` images as empty negative background samples.

- **Expected Accuracy**: Very High. The model will directly learn the visual features separating the 6 distinct packaging faces and suppress backgrounds implicitly.
- **Expected False Positives**: Low. Injecting the negative samples directly into the YOLO training set teaches the detector to output 0 boxes on non-artwork backgrounds.
- **Dataset Compatibility**: Perfect. 100% of bounding boxes map to this architecture.
- **Scalability**: High. Simple to maintain and update.
- **Deployment Complexity**: Very Low. Only one model weights file (`.pt`) to load and run in production.

## Option B: Two-Stage Pipeline
Stage 1 (Classification: Artwork vs Not_Artwork) -> Stage 2 (YOLO Detection: Panels)

- **Expected Accuracy**: Lower. Two-stage pipelines compound errors. If Stage 1 incorrectly classifies an artwork image as `not_artwork`, Stage 2 never runs, dropping recall.
- **Expected False Positives**: Moderate. Stage 1 classification models can be brittle on edge cases.
- **Dataset Compatibility**: Poor. You do not have classification labels configured, nor do you have bounding boxes for "Artwork" to train an object detector for Stage 1.
- **Scalability**: Low. Requires maintaining two separate models, two distinct datasets, and two training loops.
- **Deployment Complexity**: High. Requires a custom inference orchestrator to bridge the classifier and the detector.

## RECOMMENDED ARCHITECTURE: Option A (Single YOLO Detector)
**Justification**: Option A maximizes accuracy while minimizing engineering debt. Because "Artwork" doesn't have any spatial bounding boxes, training a Stage 1 detector for it is impossible without re-annotating the dataset from scratch. A single YOLO detector trained on the 6 panels, supplemented with the `not_artwork` negative images, natively solves the entire objective in one extremely fast, highly accurate forward pass.
