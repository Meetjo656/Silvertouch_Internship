# Training Risk Assessment

## 1. Class Imbalance Risk
**Risk Level**: HIGH
`Extra_Areas` accounts for nearly 56% of the dataset. Without mitigation, YOLO's loss function may allow the model to lazily optimize by over-predicting `Extra_Areas`. However, because `Extra_Areas` usually covers visually distinct geometry from the core panels, this spatial distinction should help YOLO differentiate them.

## 2. Overfitting Risk
**Risk Level**: MEDIUM to HIGH
We have 1476 images in the training set after augmentation, but they all stem from only 511 *original* images. 
- **YOLOv8m**: Medium risk. It has a moderate parameter count (~25.9M) and will likely generalize well if early stopping (patience) is strictly enforced.
- **YOLOv8l**: High risk. The large variant (~43.7M parameters) has immense capacity and can easily memorize 511 original images, leading to a massive gap between train and validation metrics.

## 3. Small Dataset Risk
**Risk Level**: HIGH
Training a large model (YOLOv8l) on a limited pool of 511 unique samples severely risks memorization. The structural boundaries might be simple enough to learn quickly, but a large model will overfit to the specific lighting and backgrounds of the 511 source images before it reaches its full potential.

## 4. Negative Sample Utilization
**Risk Level**: LOW (Optimal State)
The 294 `not_artwork` negative samples have been explicitly separated and dynamically injected into the splits with empty label files. This is a highly robust approach that will actively suppress false positive detections on backgrounds, heavily mitigating one of the most common object detection failures.
