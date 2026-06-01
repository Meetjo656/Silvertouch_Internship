# 🧪 Silvertouch Internship — Computer Vision Research Pipeline

> **Pharmaceutical packaging analysis** using perceptual color science, dense segmentation, and SAM-powered structural extraction.

---

## 📋 Table of Contents

<details open>
<summary><strong>Expand / Collapse</strong></summary>

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Task 1 — Perceptual Text-Color Comparison](#-task-1--perceptual-text-color-comparison)
  - [Pipeline](#task-1-pipeline)
  - [Module Breakdown](#task-1-module-breakdown)
  - [Configuration](#task-1-configuration)
  - [Running Task 1](#running-task-1)
- [Task 2 — Pharmaceutical Carton Segmentation](#-task-2--pharmaceutical-carton-segmentation)
  - [Dual-Pipeline Architecture](#dual-pipeline-architecture)
  - [Structural Pipeline](#structural-pipeline)
  - [SAM Pipeline](#sam-pipeline)
  - [Text-Removal Branch](#text-removal-branch)
  - [Module Breakdown](#task-2-module-breakdown)
  - [Configuration](#task-2-configuration)
  - [Running Task 2](#running-task-2)
- [Setup & Installation](#-setup--installation)
- [Environment Variables](#-environment-variables)
- [Output Artifacts](#-output-artifacts)
- [Thresholds & Tuning](#-thresholds--tuning)

</details>

---

## 🔭 Overview

This repository contains two independent computer vision research tasks developed during a Silvertouch Technologies internship, both targeting **pharmaceutical carton imagery**:

| | Task 1 | Task 2 |
|---|---|---|
| **Goal** | Perceptually compare text color between two crop images | Segment carton structural panels, detect & suppress text regions |
| **Core Tech** | LAB MeanShift clustering + ΔE CIE2000 | OpenCV structural segmentation + SAM (Segment Anything Model) |
| **Input** | Two cropped text images | Folder of full carton images |
| **Output** | Comparison dashboard PNG | Overlay PNGs, JSON reports, segment maps |
| **Key Metric** | ΔE ≤ 8.0 → MATCH | Coverage, IoU, mask stability |

---

## 📁 Repository Structure

```
Silvertouch Internship/
│
├── .gitignore
├── README.md
│
├── task1/                          ← Color comparison pipeline
│   ├── main.py                     ← Entry point
│   ├── config.py                   ← Thresholds & paths
│   ├── data/
│   │   ├── reference.png           ← Reference text crop
│   │   └── target.png              ← Target text crop
│   ├── outputs/
│   │   └── comparison_dashboard.png
│   ├── preprocessing/
│   │   ├── load_image.py
│   │   ├── foreground_extraction.py
│   │   └── estimate_background.py
│   ├── clustering/
│   │   └── meanshift_cluster.py
│   ├── color_processing/
│   │   ├── color_family.py
│   │   ├── deltae_compare.py
│   │   ├── dominant_color.py
│   │   └── rgb_hex.py
│   ├── visualization/
│   │   └── dashboard.py
│   └── utils/
│
└── task2/                          ← Carton segmentation pipeline
    ├── main.py                     ← Entry point (dual pipeline)
    ├── configs/
    │   ├── config.py               ← Structural config
    │   └── sam_config.py           ← SAM config
    ├── weights/
    │   └── sam_vit_h.pth           ← SAM ViT-H checkpoint (~2.4 GB)
    ├── data/
    │   ├── input/                  ← Source carton images
    │   └── output/                 ← Per-image result folders
    ├── segmentation/
    │   ├── structural_segmenter.py
    │   ├── sam_segmenter.py
    │   ├── sam_region_extractor.py
    │   ├── sam_mask_filler.py
    │   ├── mask_reconstructor.py   ← Text-removal branch
    │   ├── sam_text_classifier.py  ← Text-removal branch
    │   ├── sam_text_suppressor.py  ← Text-removal branch
    │   ├── region_analyzer.py
    │   ├── region_merger.py
    │   ├── region_splitter.py
    │   ├── rectangle_fitter.py
    │   └── segmentation_evaluator.py
    ├── geometry/
    │   ├── extract_polygon.py
    │   ├── polygon_metrics.py
    │   ├── polygon_refiner.py
    │   ├── polygon_utils.py
    │   ├── polygon_validator.py
    │   └── sam_polygon_builder.py
    ├── visualization/
    │   ├── sam_overlay_generator.py
    │   ├── polygon_overlay_generator.py
    │   ├── overlay_generator.py
    │   ├── segment_map_generator.py
    │   └── text_suppression_overlay.py
    ├── evaluation/
    │   ├── sam_segmentation_evaluator.py
    │   ├── batch_runner.py
    │   ├── batch_summary.py
    │   └── text_suppressor_evaluation.py
    ├── debug/
    │   ├── sam_debug_exporter.py
    │   ├── debug_exporter.py
    │   └── text_suppression_debug_exporter.py
    ├── models/
    │   ├── sam_mask.py
    │   └── segment.py
    ├── serialization/
    │   └── segment_serializer.py
    └── utils/
        └── loader.py
```

---

## 🎨 Task 1 — Perceptual Text-Color Comparison

Determines whether the **text color** on two pharmaceutical carton crops is **perceptually identical**, matching how the human eye — not just RGB numbers — perceives color.

### Task 1 Pipeline

```
reference.png          target.png
      │                     │
      ▼                     ▼
  load_image()          load_image()
      │                     │
      ▼                     ▼
extract_foreground()  extract_foreground()
  (ΔE from BG mask)    (ΔE from BG mask)
      │                     │
      ▼                     ▼
MeanShiftExtractor    MeanShiftExtractor
  (LAB space)           (LAB space)
      │                     │
      ▼                     ▼
get_dominant_color()  get_dominant_color()
      │                     │
      ▼                     ▼
get_color_family()    get_color_family()
get_meta_family()     get_meta_family()
      │                     │
      └──────────┬───────────┘
                 ▼
         CompareColors.is_match()
         (ΔE CIE2000 + RGB dist)
                 │
                 ▼
     generate_comparison_dashboard()
                 │
                 ▼
        outputs/comparison_dashboard.png
```

### Task 1 Module Breakdown

<details>
<summary><strong>preprocessing/</strong></summary>

| File | Class / Function | Purpose |
|---|---|---|
| `load_image.py` | `load_image()` | Read image from disk, validate shape |
| `foreground_extraction.py` | `extract_foreground()` | Build foreground mask by comparing each pixel's ΔE against estimated background |
| `estimate_background.py` | `estimate_background()` | Sample image corners to derive a representative background color in LAB space |

**Key output of `extract_foreground()`:**
```python
{
    "mask":               np.ndarray,  # binary foreground mask
    "lab":                np.ndarray,  # LAB image (foreground only)
    "delta_from_bg_values": np.ndarray # per-pixel ΔE from background
}
```

</details>

<details>
<summary><strong>clustering/</strong></summary>

| File | Class | Purpose |
|---|---|---|
| `meanshift_cluster.py` | `MeanShiftExtractor` | Runs SKLearn MeanShift on LAB foreground pixels to find density peaks |

MeanShift clusters without requiring a fixed cluster count — peaks represent the true dominant ink colors. Configured via `MEANSHIFT_QUANTILE = 0.2`.

</details>

<details>
<summary><strong>color_processing/</strong></summary>

| File | Function / Class | Purpose |
|---|---|---|
| `dominant_color.py` | `get_dominant_color()` | Select the largest cluster centroid as the ink color |
| `color_family.py` | `get_color_family()`, `get_meta_family()` | Map RGB → named color family (e.g. `"crimson"`) → meta family (e.g. `"red"`) using `colour-science` |
| `deltae_compare.py` | `CompareColors` | Compute ΔE CIE2000 between two LAB colors; threshold-based MATCH decision |
| `rgb_hex.py` | `rgb_to_hex()` | Format numpy RGB array as `#RRGGBB` |

**Match decision logic:**
```python
ΔE ≤ DELTA_E_MATCH_THRESHOLD (8.0)  →  "MATCH"
ΔE >  8.0                            →  "NO MATCH"
```

</details>

<details>
<summary><strong>visualization/</strong></summary>

| File | Function | Purpose |
|---|---|---|
| `dashboard.py` | `generate_comparison_dashboard()` | Render a multi-panel PNG showing both images, foreground masks, cluster maps, color swatches, and metrics side-by-side |

</details>

### Task 1 Configuration

```python
# task1/config.py

REFERENCE_IMAGE          = "data/reference.png"
TARGET_IMAGE             = "data/target.png"
OUTPUT_DASHBOARD         = "outputs/comparison_dashboard.png"

DELTA_E_MATCH_THRESHOLD  = 8.0    # ΔE above this → NO MATCH
FOREGROUND_DELTA_THRESHOLD = 10   # Min ΔE from BG to count as foreground
MEANSHIFT_QUANTILE       = 0.2    # Bandwidth control for MeanShift
SAVE_DEBUG_IMAGES        = False  # Toggle intermediate image saving
```

### Running Task 1

```bash
# From repo root
cd "task1"
python main.py
```

Expected console output:
```
Stage 1: Initializing Perceptual Text-Color Semantic Comparison Pipeline...
Stage 2: Loading cropped text images...
Stage 3: Extracting foreground text masks...
Stage 4: Performing density-based MeanShift clustering...
Stage 5: Selecting semantic dominant color clusters...
Stage 6: Mapping dominant colors to semantic families...
Stage 7: Generating high-fidelity Comparison Dashboard at outputs/comparison_dashboard.png...
Success: Comparative pipeline executed flawlessly.
```

---

## 🏭 Task 2 — Pharmaceutical Carton Segmentation

Segments every structural region of a pharmaceutical carton (panels, flaps, glue tabs) using **two parallel pipelines** — a classical OpenCV approach and a SAM-based deep learning approach — then compares and evaluates both.

A dedicated **text-removal branch** suppresses printed text regions from the structural mask, producing a clean `new_closed_mask` ready for downstream geometry processing.

### Dual-Pipeline Architecture

```
                    data/input/*.png
                          │
              ┌───────────┴────────────┐
              ▼                        ▼
    StructuralSegmenter          SAMSegmenter
    (OpenCV / classical)      (ViT-H checkpoint)
              │                        │
              ▼                        ▼
      Contour analysis         Raw automatic masks
      Rectangle fitting        SAMRegionExtractor
      Region labeling          SAMMaskFiller (filter)
              │                        │
              ▼                        ▼
         overlay.png            sam_overlay.png
         segment_map.png        sam_report.json
         segments.json          sam_summary.json
              │                        │
              └───────────┬────────────┘
                          ▼
                 batch_summary report
```

### Structural Pipeline

> Classical OpenCV-based segmentation: thresholding → contour detection → rectangle fitting → region labeling.

```
Image + Binary Mask
        │
        ▼
StructuralSegmenter.segment()
        │
   ┌────┴────────────────────┐
   ▼                         ▼
RegionAnalyzer          RectangleFitter
(contour stats)         (minAreaRect)
   │                         │
   ▼                         ▼
RegionMerger            Segment labels
RegionSplitter          (panel / flap / tab)
   │
   ▼
SegmentationEvaluator
   │
   ├── overlay.png
   ├── segment_map.png
   ├── report.json
   └── segments.json
```

### SAM Pipeline

> Deep learning segmentation using Meta's Segment Anything Model (ViT-H).

```
Image
  │
  ▼
SAMSegmenter.generate_masks()          ← automatic mask generation
  │
  ▼
SAMRegionExtractor.extract_regions()   ← normalise to project region dicts
  │
  ▼
SAMMaskFiller.filter_masks()           ← filter by:
  │                                        area bounds
  │                                        IoU threshold
  │                                        stability score
  │                                        geometry constraints
  │                                        duplicate removal
  ▼
SAMSegmentationEvaluator.evaluate()    ← coverage, panel/flap/tab counts
  │
  ▼
SAMOverlayGenerator                    ← overlay + IoU heatmap + stability heatmap
  │
  ▼
SAMDebugExporter                       ← debug frames, mask strips, score logs
  │
  ├── sam_overlay.png
  ├── sam_report.json
  └── sam_summary.json
```

**SAM is optional.** If `weights/sam_vit_h.pth` is absent, the structural pipeline runs normally and SAM steps are skipped with a clear message.

### Text-Removal Branch

The text-removal branch is a **self-contained sub-pipeline** that runs after SAM classification to strip printed text from the structural mask. The three core files work in sequence:

```
SAMTextClassifier          →    SAMTextSuppressor      →    MaskReconstructor
(feature extraction             (route by label)              (merge + repair)
 + labeling)
```

#### Flow in Detail

```
SAM filtered_regions
        │
        ▼
SAMTextClassifier.classify_masks()
        │
        │  compute_features() per mask:
        │    mask_area      ← foreground pixel count
        │    contour_area   ← cv2.contourArea
        │    fill_ratio     ← mask_area / bbox_area
        │    solidity       ← contour_area / hull_area
        │    stability_score← mask_area / image_size
        │    predicted_iou  ← min/union of pixel vs contour area
        │    eccentricity   ← fitEllipse axis ratio
        │    thinness_ratio ← short/long side of minAreaRect
        │    elongation     ← long/short side of minAreaRect
        │
        │  label decision:
        │    mask_area < 3000 AND fill_ratio < 0.45  →  "text"
        │    mask_area > 5000 AND fill_ratio > 0.50  →  "structure"
        │    otherwise                               →  "unknown"
        │
        ▼
[{"label": "text"|"structure"|"unknown", "features": {...}}, ...]
        │
        ▼
SAMTextSuppressor.suppress()
        │
        ├── separate_masks()   ← single routing point
        │     ├── text_masks
        │     ├── structure_masks
        │     └── unknown_masks
        │
        ├── kept_masks = structure_masks + unknown_masks
        ├── removed_masks = text_masks
        └── summary dict (counts, no string formatting)
                │
                ▼
MaskReconstructor.reconstruct(kept_masks, image_shape)
        │
        ├── draw_masks()    ← paint kept masks onto blank canvas
        ├── close_gaps()    ← MORPH_CLOSE (5×5 rect kernel)
        ├── fill_holes()    ← flood-fill inversion → OR combine
        ├── clean_noise()   ← drop components < 100 px
        └── validate_mask() ← raise if fully empty
                │
                ▼
        new_closed_mask  (uint8, values 0/1)
                │
                ▼
   Connected Components
          ↓
   Polygon Extraction
          ↓
   Region Segmentation
```

#### Text Classification Thresholds

| Feature | Text | Structure |
|---|---|---|
| `mask_area` | `< 3 000 px` | `> 5 000 px` |
| `fill_ratio` | `< 0.45` | `> 0.50` |
| Everything else | `"unknown"` (kept) | — |

> Text tends to be **small** and **sparse**. Structure tends to be **large** and **dense**.

#### MaskReconstructor — Morphological Repair

| Step | Method | Algorithm |
|---|---|---|
| 1 | `draw_masks()` | Paint `mask["mask"]` arrays onto zeros canvas |
| 2 | `close_gaps()` | `cv2.morphologyEx(MORPH_CLOSE, 5×5)` |
| 3 | `fill_holes()` | Flood-fill from border → invert → OR with mask |
| 4 | `clean_noise()` | `connectedComponentsWithStats` — drop area < 100 |
| 5 | `validate_mask()` | Raise `ValueError` if no foreground survives |

### Task 2 Module Breakdown

<details>
<summary><strong>segmentation/</strong></summary>

| File | Class | Role |
|---|---|---|
| `structural_segmenter.py` | `StructuralSegmenter` | Classical OpenCV full pipeline |
| `sam_segmenter.py` | `SAMSegmenter` | SAM checkpoint loader + automatic mask generator |
| `sam_region_extractor.py` | `SAMRegionExtractor` | Normalise SAM dicts → project region format |
| `sam_mask_filler.py` | `SAMMaskFiller` | Multi-criteria mask filtering |
| `region_analyzer.py` | `RegionAnalyzer` | Contour statistics, aspect ratios, centroids |
| `region_merger.py` | `RegionMerger` | Merge adjacent/overlapping regions |
| `region_splitter.py` | `RegionSplitter` | Split over-merged regions |
| `rectangle_fitter.py` | `RectangleFitter` | Fit minimum-area rectangles to contours |
| `segmentation_evaluator.py` | `SegmentationEvaluator` | IoU, coverage, classification metrics |
| `sam_text_classifier.py` | `SAMTextClassifier` | Feature extraction + text/structure labeling |
| `sam_text_suppressor.py` | `SAMTextSuppressor` | Separate & suppress text masks |
| `mask_reconstructor.py` | `MaskReconstructor` | Merge → close → fill → clean → validate |

</details>

<details>
<summary><strong>geometry/</strong></summary>

| File | Class | Role |
|---|---|---|
| `extract_polygon.py` | — | Contour → polygon vertices |
| `polygon_metrics.py` | — | Area, perimeter, compactness |
| `polygon_refiner.py` | `PolygonRefiner` | Smooth and simplify polygon vertices |
| `polygon_utils.py` | — | Shared geometry helpers |
| `polygon_validator.py` | `PolygonValidator` | Self-intersection checks, minimum vertex count |
| `sam_polygon_builder.py` | `SAMPolygonBuilder` | Build polygons from SAM mask contours |

</details>

<details>
<summary><strong>visualization/</strong></summary>

| File | Class | Role |
|---|---|---|
| `sam_overlay_generator.py` | `SAMOverlayGenerator` | Coloured mask overlays, IoU heatmap, stability heatmap |
| `polygon_overlay_generator.py` | `PolygonOverlayGenerator` | Draw polygon outlines on image |
| `overlay_generator.py` | — | Generic structural overlay |
| `segment_map_generator.py` | — | False-colour segment map |
| `text_suppression_overlay.py` | — | Green = kept, Red = removed text |

</details>

<details>
<summary><strong>evaluation/</strong></summary>

| File | Class | Role |
|---|---|---|
| `sam_segmentation_evaluator.py` | `SAMSegmentationEvaluator` | Coverage, panel/flap/tab counts, status, JSON report |
| `batch_runner.py` | `BatchRunner` | Run pipeline over all images |
| `batch_summary.py` | `BatchSummary` | Aggregate per-image results, console + file report |
| `text_suppressor_evaluation.py` | — | Evaluate text suppression quality |

</details>

<details>
<summary><strong>debug/</strong></summary>

| File | Class | Role |
|---|---|---|
| `sam_debug_exporter.py` | `SAMDebugExporter` | Export raw/filtered frames, mask strips, score logs |
| `debug_exporter.py` | `DebugExporter` | Export structural pipeline intermediates |
| `text_suppression_debug_exporter.py` | — | Export text-removal intermediates |

</details>

<details>
<summary><strong>models/</strong></summary>

| File | Class | Role |
|---|---|---|
| `sam_mask.py` | `SAMMask` | Typed wrapper for a single SAM mask dict |
| `segment.py` | `Segment` | Typed representation of a labeled carton segment |

</details>

### Task 2 Configuration

<details>
<summary><strong>configs/config.py</strong> — Structural pipeline</summary>

```python
config = {
    "input_dir":  "data/input",
    "output_dir": "data/output",
    # thresholds set inside config.py
}
```

</details>

<details>
<summary><strong>configs/sam_config.py</strong> — SAM pipeline</summary>

```python
sam_config = {
    # Mask generator parameters
    "points_per_side":        32,
    "pred_iou_thresh":        0.88,
    "stability_score_thresh": 0.95,
    "min_mask_region_area":   500,

    # Filtering bounds
    "min_area_ratio":   0.005,
    "max_area_ratio":   0.95,

    # Geometry constraints
    "min_aspect_ratio": 0.1,
    "max_aspect_ratio": 10.0,
}
```

</details>

### Running Task 2

```bash
cd "task2"

# With SAM (checkpoint must exist at weights/sam_vit_h.pth)
python main.py

# Without SAM (structural pipeline only)
# Just run the same command — SAM is auto-skipped if checkpoint is missing
python main.py

# Override checkpoint path
set SAM_CHECKPOINT=C:\path\to\sam_vit_h_4b8939.pth
python main.py
```

**Expected console output (per image):**
```
============================================================
  Processing: carton_01.png
============================================================
  [Structural] segments=12  rectangles=10
  [SAM] raw= 87  filtered= 23
  [SAM] status=good  P=4  F=6  T=3  coverage=0.91
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd "Silvertouch Internship"

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS / Linux

# 3. Install dependencies
pip install opencv-python numpy scikit-learn colour-science

# 4. (Task 2 SAM only) Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install torch torchvision
```

**Python version:** 3.10+ required (uses `X | Y` type union syntax and `match` statements).

---

## 🌍 Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `SAM_CHECKPOINT` | `task2/weights/sam_vit_h.pth` | Path to SAM ViT-H weights file |

---

## 📦 Output Artifacts

### Task 1

| File | Contents |
|---|---|
| `outputs/comparison_dashboard.png` | Side-by-side panel: images, masks, cluster maps, swatches, ΔE, match status |

### Task 2 (per image, inside `data/output/<image_name>/`)

| File | Contents |
|---|---|
| `overlay.png` | Structural pipeline — coloured region overlay |
| `segment_map.png` | False-colour segment map |
| `segments.json` | Labeled segments (panel, flap, tab) with bounding boxes |
| `report.json` | Structural evaluation report |
| `sam_overlay.png` | SAM coloured mask overlay |
| `sam_report.json` | Full SAM evaluation (coverage, counts, status) |
| `sam_summary.json` | Compact SAM summary (raw/filtered counts) |
| `debug/` | Intermediate debug frames from both pipelines |

---

## 🎛️ Thresholds & Tuning

### Task 1

| Parameter | Default | Effect |
|---|---|---|
| `DELTA_E_MATCH_THRESHOLD` | `8.0` | Raise → more lenient color matching |
| `FOREGROUND_DELTA_THRESHOLD` | `10` | Raise → stricter foreground extraction |
| `MEANSHIFT_QUANTILE` | `0.2` | Lower → finer clusters; raise → fewer, broader clusters |

### Task 2 — Text Classifier

| Feature | Text threshold | Structure threshold | Rationale |
|---|---|---|---|
| `mask_area` | `< 3 000 px` | `> 5 000 px` | Text is small |
| `fill_ratio` | `< 0.45` | `> 0.50` | Text is sparse within bbox |

### Task 2 — MaskReconstructor

| Parameter | Default | Effect |
|---|---|---|
| `CLOSE_KERNEL_SIZE` | `(5, 5)` | Larger → bridges wider text-removal gaps |
| `MIN_COMPONENT_AREA` | `100 px` | Raise → more aggressive noise removal |

---

<div align="center">

Built during a Computer Vision internship at **Silvertouch Technologies**  
Python · OpenCV · NumPy · scikit-learn · colour-science · SAM (Meta AI)

</div>
