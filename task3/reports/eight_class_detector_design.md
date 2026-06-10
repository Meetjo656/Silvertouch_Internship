# 8-Class Detector Design Proposal

To satisfy the requirement of an 8-class detector, we must synthetically generate bounding boxes for `Artwork` and `Not_Artwork` because none exist in the raw dataset.

## Proposed Labeling Strategy

Since `Artwork` and `Not_Artwork` represent global image classifications rather than localized geometric features, the correct approach in a YOLO framework is to use **full-image bounding boxes**.

### 1. Artwork Images (Structural Dataset)
Every image in the current `train`, `test`, and `valid` splits will receive a synthetic bounding box covering the entire image:
* **Class:** `Artwork`
* **YOLO Format:** `<class_id> 0.5 0.5 1.0 1.0` (Center X, Center Y, Width, Height)
This box will perfectly overlap the entire image, coexisting with the local structural panel boxes (Front, Side, etc.) nested inside it.

### 2. Not_Artwork Images (Scraped Dataset)
Currently, scraped images are fed to YOLO as "negative samples" (images with no corresponding `.txt` file or an empty `.txt` file).
We will redesign this: every scraped image will now receive a `.txt` file containing a single full-image bounding box.
* **Class:** `Not_Artwork`
* **YOLO Format:** `<class_id> 0.5 0.5 1.0 1.0`

## Impact Evaluation

### Dataset Impact
* Adds `~1476` full-image `Artwork` labels.
* Adds `294` full-image `Not_Artwork` labels.

### Class Imbalance Impact
These classes will have very high representation. However, because they are global macro-features (entire images), they will not directly compete with the localized micro-features (flaps and panels).

### Training Impact
The YOLO classification head will learn global texture, typography presence, and overall composition to distinguish the full-image `Artwork` boxes from the `Not_Artwork` boxes. 

### Inference Impact
When passing an image to the model, it will draw a massive bounding box around the entire perimeter labeled `Artwork` (or `Not_Artwork`), while simultaneously drawing the smaller structural boundaries inside.
