# Final Dataset Architecture Decision

## 1. Actual Annotation Hierarchy
- **Parent/Taxonomy Nodes**: `Artwork`, `not_artwork`
- **Child/Spatial Nodes**: `Front_Panel`, `Side_Panel`, `Back_Panel`, `Top_Panel`, `Bottom_Part`, `Extra_Areas`

## 2. Actual Trainable Classes
There are exactly **6 trainable spatial classes**:
1. Front_Panel
2. Side_Panel
3. Back_Panel
4. Top_Panel
5. Bottom_Part
6. Extra_Areas

## 3. Should Artwork exist in YOLO?
**NO.** 
YOLO is an object detection framework that requires bounding boxes to train. Because `Artwork` has zero bounding boxes in the JSON, it is physically impossible to map it to a YOLO class.

## 4. Should not_artwork exist in YOLO?
**YES, but as negative backgrounds, not as a class label.**
The images exist, but they have zero bounding boxes. In YOLO, images with empty label files (`.txt` files with 0 bytes) are treated as "Negative Samples". This is a core YOLO feature that drastically reduces false positives without requiring a dedicated class ID.

## 5. Recommended Production Architecture
**Single-Stage YOLO Object Detector.**
Pass all images directly into the detector. It will natively learn to find the 6 panels on artwork images, and learn to find nothing on non-artwork backgrounds.

## 6. Recommended Final YOLO Class List
1. `Front_Panel`
2. `Side_Panel`
3. `Back_Panel`
4. `Top_Panel`
5. `Bottom_Part`
6. `Extra_Areas`
*(Note: A 7th implicit "Background" class exists internally within YOLO's loss function).*

## 7. Recommended Training Roadmap
- **Dataset Phase**: Do not regenerate or modify the pipeline yet.
- **Next Phase**: Update `category_map` logic in the pipeline to exclusively map the 6 structural classes above (mapping them to IDs 0-5), while generating empty `.txt` files for all `not_artwork` images.
