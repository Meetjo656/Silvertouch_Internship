# Dataset Audit Report

## Overall Dataset Size
- **Total Original Images Available**: 511
  - Artwork Images (train + valid + test): 217
  - Scrapped / Negative Samples: 294
- **Previously Used Original Images**: 477 *(The previous script ignored the 34 images in `valid` and `test`)*
- **Augmented Images Generated**: ~999
- **Total Images After Augmentation**: 1476

## Annotation Statistics (Across Train/Valid/Test)
- **Total Annotations**: 8,917
- **Total Bounding Box Annotations**: 8,917
- **Total Segmentation Annotations**: 56
- **Number of Classes**: 20
- **Average Objects Per Image**: ~41.0 objects/image
- **Average Polygon Vertices**: ~7.2 vertices (for the few existing polygons)

## Data Format Coverage
- **Percentage of Annotations With Valid Segmentation**: **0.63%**
- **Percentage of Annotations With Bounding Boxes Only**: **99.37%**

## Summary
The dataset is extremely dense with objects (~41 per image) but is exclusively a bounding box dataset. Polygon coverage is virtually non-existent.
