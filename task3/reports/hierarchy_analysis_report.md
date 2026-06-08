# Hierarchy Validation

## Analysis
The `_annotations_mapped.json` defines "Artwork" as Category ID 1. However, an exhaustive scan of the 2,879 bounding box annotations across all images reveals **0 instances** of the "Artwork" class. 

Every single bounding box in the dataset points exclusively to a specific child structural panel (`Front_Panel`, `Side_Panel`, etc.).

## Conclusion
"Artwork" is exclusively a parent taxonomy concept. It is not an independent visual class in your annotations.

### Does Artwork contain its own annotations?
**NO.**
