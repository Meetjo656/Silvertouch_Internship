# Hierarchy Validation Report

## Annotation Audit
Based on the `_annotations_mapped.json` files across the `train`, `valid`, and `test` splits:

- Total annotations for category_id=1 (Artwork): **0**
- Total annotations for category_id=2 (not_artwork): **0**
- Total annotations for category_id=3..8 (Structural Panels): **2,879**

## Answers to Architectural Questions

**Is Artwork a real detection class?**
**NO**
There are exactly zero bounding box annotations for "Artwork" in the entire dataset. It functions solely as a parent organizational node.

**Is not_artwork a real detection class?**
**NO**
There are exactly zero bounding box annotations for "not_artwork". It is an image-level classification label indicating background/negative sample images, not a localized bounding box target.

**Are Front/Side/Back/Top/Bottom/Extra the only structural object classes?**
**YES**
100% of the 2,879 spatial annotations in this dataset belong to these 6 structural categories.
