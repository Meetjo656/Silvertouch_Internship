# Model Selection Report

## Dataset Characteristics
- **Total Images**: Small (511 total, only 217 contain objects).
- **Annotation Density**: Extremely High (~41 objects per image).
- **Classes**: 20 classes.
- **Overfitting Risk**: Severe. A high parameter model will easily memorize 183 training images.

## Model Comparisons
| Model | Parameters | Speed | Analysis for this Dataset |
|---|---|---|---|
| **YOLOv8n** | 3.2M | Fastest | May struggle to capture the complex relationships of 20 classes tightly packed at 41 objects/image. |
| **YOLOv8s** | 11.2M | Fast | Good baseline, but might underfit the dense spatial relationships. |
| **YOLOv8m** | 25.9M | Medium | **Optimal Balance**. High enough capacity to resolve 20 classes and dense overlaps, but small enough to resist memorizing a tiny 183-image dataset. |
| **YOLOv8l** | 43.7M | Slow | Overkill. Very high risk of severe overfitting on only 183 source images. |
| **YOLOv8x** | 68.2M | Slowest | Extreme overfitting guaranteed. Will memorize the training set rather than generalize. |

## Ranked Recommendation
1. **YOLOv8m**: Best probability of high accuracy without extreme overfitting. 
2. **YOLOv8s**: Safer choice if YOLOv8m begins to overfit early in training.
3. **YOLOv8l**: (Current choice). Only use if you apply extremely heavy augmentation and strict early stopping.

**Recommendation:** Start with **YOLOv8m**. It has the structural capacity to handle dense object detection (41/image) but mitigates the risk of overfitting on such a small dataset.
