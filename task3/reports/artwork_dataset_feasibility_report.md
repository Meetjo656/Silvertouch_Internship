# Dataset Feasibility Analysis: Hidden Annotations Audit

This report investigates the raw COCO datasets (`_annotations_mapped.json`) to determine if `Artwork` and `Not_Artwork` annotations were accidentally filtered out, or if they simply do not exist.

## Raw COCO Annotation Scan Results

A deep recursive scan of `train`, `test`, and `valid` JSON files was performed looking for:
* **Category ID 1** (`Artwork`)
* **Category ID 2** (`not_artwork`)

### Findings:
* **Artwork Annotations Count**: `0`
* **Not_Artwork Annotations Count**: `0`

## Conclusion

**There are no hidden annotations.** The classes were not incorrectly filtered by our scripts. 

The original annotator on Roboflow created the *category labels* in the project schema, but they never actually drew a single bounding box assigned to those categories on any image. They only drew polygons for the 6 structural faces. 

Therefore, if we must include these classes in the training pipeline, **we must synthetically generate the bounding boxes ourselves.**
