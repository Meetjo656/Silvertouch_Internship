# Training Strategy Review

### Evaluate: YOLOv8m vs YOLOv8l

#### YOLOv8m
- **Expected Accuracy**: High. Excellent balance for densely packed datasets.
- **Expected Generalization**: Strong. Less likely to memorize the ~200 source images.
- **Overfitting Risk**: Moderate. Still requires `patience=50`.
- **Training Cost**: Medium. Fast enough for rapid iteration.

#### YOLOv8l
- **Expected Accuracy**: Potentially high, but often drops below YOLOv8m on small datasets due to overfitting.
- **Expected Generalization**: Weak. Very high risk of memorizing the exact bounding boxes of the training set.
- **Overfitting Risk**: Severe.
- **Training Cost**: High. Slower iteration speeds.

**Conclusion**: YOLOv8m must be your primary benchmark. Only promote to YOLOv8l if YOLOv8m shows underfitting (e.g. failing to capture specific classes despite high image size).
