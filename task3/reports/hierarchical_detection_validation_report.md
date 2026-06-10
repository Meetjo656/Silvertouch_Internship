# Hierarchical Detection Validation Report

This report validates the implementation of a 2-level hierarchical detection layer built on top of the 6-class structural detector without retraining.

## Validation Results

| Image Name | Ground Truth | Prediction | Detections | Highest Confidence | Status |
|---|---|---|---|---|---|
| `label2_13.jpg` | Artwork | **Artwork** | 1 | 0.337 | **PASS** |
| `back.jpg` | Artwork | **Not_Artwork** | 0 | 0.000 | **FAIL** |
| `20260115-S_IO_jpg.rf.f4fdb49b9a94164cd70aab09ea3572d1.jpg` | Artwork | **Artwork** | 13 | 0.993 | **PASS** |
| `Label3.jpg` | Artwork | **Not_Artwork** | 0 | 0.000 | **FAIL** |
| `bag1.jpg` | Artwork | **Not_Artwork** | 0 | 0.000 | **FAIL** |
| `59316-208 front label.jpg` | Not_Artwork | **Not_Artwork** | 0 | 0.000 | **PASS** |
| `5258cef2-0e5c-4ccd-e063-6394a90a9e5c_image_001.jpg` | Not_Artwork | **Artwork** | 1 | 0.265 | **FAIL** |
| `484b0833-9f8a-f155-e063-6294a90a1869_image_002.jpg` | Not_Artwork | **Not_Artwork** | 0 | 0.000 | **PASS** |
| `343531 Folding Carton.jpg` | Not_Artwork | **Not_Artwork** | 0 | 0.000 | **PASS** |
| `EQ52570A.jpg` | Not_Artwork | **Not_Artwork** | 0 | 0.000 | **PASS** |

## Output Format Demonstration
The inference function now natively returns the hierarchical JSON structure:
```json
{
    "image_type": "Artwork",
    "detections": [
        {
            "class": "Extra_Areas",
            "confidence": 0.3371706008911133,
            "bbox": [
                38.492164611816406,
                527.2610473632812,
                543.2286987304688,
                638.909912109375
            ]
        }
    ]
}
```