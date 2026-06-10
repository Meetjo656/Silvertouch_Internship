# Hierarchical Detection Compatibility Analysis

This document evaluates the mathematical and architectural compatibility of YOLOv8 when dealing with overlapping global and local classes within the same image.

## YOLOv8 Capabilities

### 1. Nested Boxes (Full-image vs. Local)
**Compatible.** YOLOv8 treats every bounding box as an independent detection object. It is entirely capable of predicting a box that spans `1.0` width and `1.0` height (the `Artwork` class) while simultaneously predicting a box that spans `0.2` width (the `Front_Panel` class) completely nested inside of it.

### 2. Multi-Scale Detection Effects
**Compatible.** YOLO utilizes a Feature Pyramid Network (FPN). 
* The full-image `Artwork` box will be detected by the deepest, coarsest grid layers (designed for large objects).
* The structural panels will be detected by the shallower, high-resolution grid layers (designed for medium/small objects).
They do not compete for the same anchor or grid-cell assignments.

### 3. NMS (Non-Maximum Suppression) Side Effects
**Compatible.** NMS is applied on a *per-class* basis. 
Because `Artwork` is a completely different class ID than `Front_Panel`, YOLO will **never** suppress the structural panels just because they are heavily overlapped by the `Artwork` box. NMS will only suppress duplicate overlapping boxes of the exact same class.

### 4. Parent-Child Limitations
**Limitation.** YOLO does not inherently understand hierarchy. It will not logically enforce that a `Front_Panel` *must* be inside an `Artwork` box. It will predict both independently based on pixel features. However, for visual demonstration purposes, this will still fulfill the requirement successfully.
