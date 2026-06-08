# Final Class Recommendation

Based on the exhaustive audit of the `_annotations_mapped.json` dataset, **Scenario A** is the undeniable reality.

## Findings
The class "Artwork" has 0 bounding boxes. Including an empty class that never appears in the training data will needlessly consume model parameters and could confuse precision/recall metrics.

## Recommended YOLO Classes
You must train on exactly **7 classes**:
1. `not_artwork` (used for the negative samples)
2. `Front_Panel`
3. `Side_Panel`
4. `Back_Panel`
5. `Top_Panel`
6. `Bottom_Part`
7. `Extra_Areas`

*(Note: The actual YOLO class ID mapping will be defined sequentially, dropping the unused `Artwork` class).*

## Justification
Because "Artwork" is just an organizational taxonomy node and has no bounding boxes, trying to train the model to predict it is impossible. The model will instead learn to perfectly detect the 6 structural faces of the packaging, plus the `not_artwork` negative backgrounds.
