# 🧪 Silvertouch Internship — Computer Vision & Machine Learning Pipeline

> **Pharmaceutical packaging analysis** using perceptual color science, geometry-first structural layout reconstruction, deep segmentation models, and custom data scraping.

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
- [Task 2 — Structural Packaging Layout Reconstruction](#-task-2--structural-packaging-layout-reconstruction)
  - [Architecture & Pipelines](#task-2-architecture--pipelines)
  - [Geometric Feature Extraction](#geometric-feature-extraction)
  - [Model Classifier](#model-classifier)
  - [Module Breakdown](#task-2-module-breakdown)
  - [Running Task 2](#running-task-2)
- [Task 3 — DailyMed Scraper & Dataset Curation](#-task-3--dailymed-scraper--dataset-curation)
  - [DailyMed Artwork Scraper](#dailymed-artwork-scraper)
  - [Roboflow Dataset Curation](#roboflow-dataset-curation)
  - [Module Breakdown](#task-3-module-breakdown)
  - [Running Task 3](#running-task-3)
- [Task 4 — Future Workspace](#-task-4--future-workspace)
- [Setup & Installation](#-setup--installation)

</details>

---

## 🔭 Overview

This repository contains computer vision and machine learning pipelines developed during a Silvertouch Technologies internship, focusing on **pharmaceutical packaging design and structural analysis**:

| Component | [Task 1](file:///mnt/d/Silvertouch%20Internship/task1) | [Task 2 (Neural Network)](file:///mnt/d/Silvertouch%20Internship/task2_neural_network) | [Task 3](file:///mnt/d/Silvertouch%20Internship/task3) |
|---|---|---|---|
| **Goal** | Perceptually compare text color between two crop images. | Reconstruct packaging structural layout (panels, flaps, glue tabs) using geometry-first PSLGs and classify masks using geometric features. | Build packaging artwork scraper and curate/normalize annotations. |
| **Core Tech** | LAB MeanShift clustering + ΔE CIE2000 | PSLG builder + Polygonizer + Random Forest classification (scikit-learn) | Multi-threaded Scraper + PIL Lanczos resizing + COCO label mapping |
| **Input** | Two cropped text images. | Original carton images + candidate segment masks. | DailyMed SETIDs / Roboflow dataset annotated in COCO format. |
| **Output** | Comparison dashboard PNG. | GeoJSON partitions, boundary overlay PNGs, trained classifier model. | Deduplicated artwork, resized images, unified COCO annotations. |
| **Key Metric** | ΔE ≤ 8.0 → MATCH | Coverage %, Overlap %, classification accuracy. | Scrape speed, deduplication rate, resolution consistency. |

---

## 📁 Repository Structure

```
Silvertouch Internship/
│
├── .gitignore
├── README.md
│
├── src/                                ← Shared Utilities
│   └── visualization/
│       └── face_boundary_renderer.py   ← Renders recovered polygon faces as boundaries onto images
│
├── task1/                              ← Text Color Comparison Pipeline
│   ├── main.py                         ← Entry point
│   ├── config.py                       ← Thresholds & paths
│   ├── data/
│   │   ├── reference.png
│   │   └── target.png
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
├── task2_neural_network/               ← Structural Layout Reconstruction & Machine Learning Pipeline
│   ├── generate_and_ingest.py          ← Ingests images and SAM masks
│   ├── analyze_masks.py                ← Computes area histograms & cumulative coverage stats
│   ├── requirements.txt                ← Dependencies
│   ├── config/                         ← Configuration files
│   ├── data/                           ← Data storage
│   │   ├── raw/                        ← Source carton images
│   │   ├── raw_temp/                   ← Temporary processing storage
│   │   ├── interim/                    ← Intermediate outputs (GeoJSONs, CSV stats)
│   │   ├── metadata/                   ← JSON dataset metadata
│   │   ├── processed/                  ← Final processed datasets
│   │   └── output_segments/
│   ├── models/                         ← Trained models (.joblib outputs)
│   ├── notebooks/                      ← Jupyter Notebooks
│   ├── tests/                          ← Unit tests
│   └── src/                            ← Core Source Code
│       ├── analysis/                   ← Audits (Outer boundary, ambiguity, overlap, topology)
│       ├── blueprint/                  ← Planar partitioning, validation, filters (Containment, Tiny face)
│       ├── core/                       ← Shared data structures & enums (labels.py)
│       ├── data/                       ← Dataset building & SAM pipeline ingestion
│       ├── features/                   ← Geometric feature extraction (geometric.py)
│       ├── geometry/                   ← Line detection/regularization, PSLG construction, polygonizer, artwork suppressor
│       ├── models/                     ← Baseline models (Random Forest classifier wrapper)
│       ├── pipeline/                   ← Orchestration (Geometry pipeline, face cleanup pipeline)
│       ├── tools/                      ← Audits, labeling tools (Label GUI, reset labels, candidate filter)
│       ├── training/                   ← Model training scripts
│       └── utils/                      ← Shared helpers & loggers
│
├── task3/                              ← DailyMed Scraper & Roboflow Dataset Curation
│   ├── README.dataset.txt              ← Roboflow dataset information
│   ├── README.roboflow.txt             ← Dataset statistics (217 images, 640x640 resize, horizontal flip)
│   ├── combine_eval.py                 ← Combines test and validation splits with mapped category IDs
│   ├── migrate_test.py                 ← Maps test set annotation coordinates/labels
│   ├── migrate_valid.py                ← Maps validation set annotation coordinates/labels
│   ├── resize_dataset.py               ← Standardizes image sizes to max 1000px using Lanczos downsampling
│   ├── train/                          ← Dataset split for training
│   ├── test/                           ← Dataset split for testing
│   ├── valid/                          ← Dataset split for validation
│   └── Scrapped/                       ← DailyMed packaging artwork scraper
│       ├── README.md
│       ├── requirements.txt
│       ├── output/
│       │   ├── images/                 ← Downloaded packaging artwork
│       │   ├── metadata/               ← Metadata outputs (images.csv, duplicates.csv)
│       │   └── logs/                   ← Detailed logging logs
│       └── src/                        ← Python crawler, downloader, filters, parser, and main script
│
└── task4/                              ← Blank directory created for future tasks
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

## 🏭 Task 2 — Structural Packaging Layout Reconstruction

Reconstructs the structural panels, dust flaps, and glue tabs of a pharmaceutical carton using **two parallel systems**: a geometry-first planar partition pipeline (line detection and graph-based polygonization) and a machine learning feature-classification pipeline (Random Forest models evaluating shape geometry).

### Architecture & Pipelines

1. **Geometry-First Partition Pipeline**:
   ```
   Original Image
         │
         ▼
     Stage 0: Artwork Suppression (Bilateral + Opening)
         │
         ▼
     Stage 1: Edge Recovery (Canny + Sobel)
         │
         ▼
     Stage 2: Line Detection & Regularization (Hough Transformations)
         │
         ▼
     Stage 3: PSLG Construction & Planar Face Polygonization
         │
         ▼
     Stage 4: Planar Partition Building (Shared borders, closed polygons)
         │
         ▼
     Outputs: GeoJSON Face Boundary polygons & visual _blueprint_final.png
   ```

2. **Cleanup & Auditing Pipeline (Phase 7B-Lite)**:
   Performs topological filtering and structural region consolidation:
   - **FaceCleanupAudit**: Verifies polygon coordinates against image dimensions.
   - **ContainmentFilter**: Eliminates nested and duplicate containment loops.
   - **TinyFaceFilter**: Rejects micro-polygons (noise generated around labels or small text artifacts).
   - **StructuralRegionMerger**: Consolidates overlapping and adjacent shapes into clean structural regions.
   - **StructuralConnectivityValidator**: Confirms final boundary connectivity.

### Geometric Feature Extraction

`extract_geometric_features()` processes binary segment masks (such as those output by SAM) to extract a complete 17-dimensional geometric profile:

* **Basic Shape Properties**: `area`, `relative_area`, `perimeter`.
* **Bounding Attributes**: `bbox_width`, `bbox_height`, `aspect_ratio`.
* **Topological Complexity**: `solidity` (contour area vs convex hull area), `extent` (contour area vs bounding box area), `vertex_count` (Douglas-Peucker simplification).
* **Circular Properties**: `equiv_diameter`, `circularity`.
* **Rotation-Invariant Properties**: rotated width/height (`rotated_width`, `rotated_height`), rotation-invariant aspect ratio (`rotated_aspect_ratio`), and minimum enclosing rectangle area (`rotated_area`).
* **Spatial Centroids**: `centroid_x`, `centroid_y` moments.

### Model Classifier

`StructuralMaskRFClassifier` wraps a scikit-learn `RandomForestClassifier` trained on the geometric features of candidate masks.
* **Leakage Prevention**: Employs `GroupShuffleSplit` (80/20 grouped by image ID) to guarantee that masks from the same carton are never mixed between training and validation sets.
* **Structural Labels** (defined in `labels.py`):
  * `PANEL`: Principal structural faces of the carton (Front, Back, Side, Top, Bottom).
  * `FLAP`: Closing dust flaps.
  * `GLUE_TAB`: Connection flaps.
  * `DUST_FLAP`: Dust flaps.
  * `BACKGROUND` / `UNKNOWN`.

### Task 2 Module Breakdown

<details>
<summary><strong>pipeline/ & blueprint/</strong></summary>

| File | Class | Role |
|---|---|---|
| `geometry_pipeline.py` | `run_geometry_pipeline()` | Runs full line detection and planar straight line graph polygonizer |
| `phase_7b_lite_pipeline.py` | `run_phase_7b_lite_pipeline()` | Consolidation, filtering, and validation of faces |
| `containment_filter.py` | `ContainmentFilter` | Filters out duplicate nested containment polygons |
| `tiny_face_filter.py` | `TinyFaceFilter` | Removes micro-polygons representing background noise |
| `structural_region_merger.py` | `StructuralRegionMerger` | Merges fragmented polygon structures |

</details>

<details>
<summary><strong>geometry/</strong></summary>

| File | Class | Role |
|---|---|---|
| `artwork_suppressor.py` | `ArtworkSuppressor` | Suppresses high-contrast graphics and text using morphological openings |
| `edge_detector.py` | `EdgeDetector` | Extracts sharp carton edges |
| `line_detector.py` | `LineDetector` | Executes Hough line transforms |
| `line_regularizer.py` | `LineRegularizer` | Groups, snaps, and straightens lines |
| `pslg_builder.py` | `PSLGBuilder` | Generates planar straight-line graphs |
| `polygonizer.py` | `Polygonizer` | Extracts closed polygons from graph edges |

</details>

<details>
<summary><strong>features/ & models/ & training/</strong></summary>

| File | Class | Role |
|---|---|---|
| `geometric.py` | `extract_geometric_features` | Calculates 17 geometric shape features from binary masks |
| `rf_classifier.py` | `StructuralMaskRFClassifier` | Scikit-learn RandomForest pipeline wrapper |
| `train.py` | `train_model()` | Loads datasets, extracts features, trains classifier, and dumps model |
| `analyze_masks.py` | `analyze()` | Performs dataset-wide coverage and cumulative area analysis |
| `generate_and_ingest.py` | `main()` | Auto-generates SAM masks and registers them into dataset metadata |

</details>

### Running Task 2

```bash
cd "task2_neural_network"

# 1. Run mask analysis (creates histograms & outputs coverage logs)
python analyze_masks.py

# 2. Train the Random Forest geometric classifier
python -m src.training.train --project_root . --n_estimators 500

# 3. Run the geometry-first PSLG polygonization pipeline
python -m src.pipeline.geometry_pipeline

# 4. Clean up faces and render structural layout blueprints
python -m src.pipeline.phase_7b_lite_pipeline
```

---

## 📥 Task 3 — DailyMed Scraper & Dataset Curation

A two-part data engineering pipeline designed to collect raw pharmaceutical packaging graphics from DailyMed and standardize annotated image datasets for downstream segmentation.

### DailyMed Artwork Scraper

Located in the [task3/Scrapped](file:///mnt/d/Silvertouch%20Internship/task3/Scrapped) directory, this pipeline crawls and downloads packaging artwork images using multi-threading and duplicate detection:

* **Robust Multi-threading**: Rapid parallel image downloads.
* **Perceptual Image Hashing**: Deduplication using `imagehash` (perceptual hashing) to automatically flag duplicate uploads.
* **Logo Filtering**: Rejects tiny files, aspect-ratio outliers, logos, and UI graphics.
* **Interrupted Session Resumption**: Tracks downloaded SETIDs and resumes seamlessly.

### Roboflow Dataset Curation

Processes and maps inconsistent Roboflow image annotations (COCO format) into standard structural partitions:

1. **Resolution Normalization**:
   - `resize_dataset.py` uses PIL's Lanczos filter to downsample high-resolution packaging images to a maximum size of 1000px, reducing model memory footprints.
2. **COCO Annotation Mapping**:
   - `migrate_test.py` and `migrate_valid.py` map raw category IDs from different label exports into unified IDs:
     * `1` → `Artwork`
     * `2` → `not_artwork`
     * `3` → `Front_Panel`
     * `4` → `Side_Panel`
     * `5` → `Back_Panel`
     * `6` → `Extra_Areas`
     * `7` → `Bottom_Part`
     * `8` → `Top_Panel`
3. **Split Consolidation**:
   - `combine_eval.py` merges test and validation splits to compile a standardized evaluation dataset (`eval_combined`).

### Task 3 Module Breakdown

<details>
<summary><strong>Scrapped/src/ (DailyMed Scraper)</strong></summary>

| File | Module | Role |
|---|---|---|
| `main.py` | Main entry point | Configures scraping loops, reads SETIDs, and drives downloads |
| `crawler.py` | DailyMed Crawler | Crawls product XML pages to find artwork image URLs |
| `downloader.py` | Multi-threaded Downloader | Handles requests, saves to disk, and tracks downloads |
| `filters.py` | Size & Format Filter | Filters out thumbnails and non-artwork logos |
| `parser.py` | HTML XML Parser | Extracts image metadata and media nodes |
| `utils.py` | Hashing & Deduplicator | Generates perceptual hashes (`phash`) to flag duplicate shapes |

</details>

<details>
<summary><strong>Dataset Curation Scripts</strong></summary>

| File | Purpose |
|---|---|
| `resize_dataset.py` | Downsamples training, validation, and scrapped image paths to a maximum resolution of 1000px |
| `combine_eval.py` | Merges test and validation splits, mapping COCO categories to standard panels |
| `migrate_test.py` / `migrate_valid.py` | Individual category mapping scripts for dataset splits |

</details>

### Running Task 3

```bash
cd "task3"

# 1. Run the DailyMed artwork scraper
python Scrapped/src/main.py

# 2. Downsample dataset to a standard maximum size of 1000px
python resize_dataset.py

# 3. Standardize and map COCO annotation categories for splits
python migrate_test.py
python migrate_valid.py

# 4. Combine test and validation splits for unified evaluation
python combine_eval.py
```

---

## 🔭 Task 4 — Future Workspace

The [task4](file:///mnt/d/Silvertouch%20Internship/task4) directory is a clean workspace initialized for future computer vision research tasks, modeling pipelines, or dataset curation scripts.

---

## ⚙️ Setup & Installation

### Requirements

- **Python Version**: 3.11+
- **Core Dependencies**: `opencv-python`, `numpy`, `scikit-learn`, `colour-science`, `scikit-image`, `pandas`, `shapely`, `matplotlib`, `seaborn`, `joblib`, `ImageHash`

### Virtual Environment Setup

```bash
# Clone the repository
git clone <repo-url>
cd "Silvertouch Internship"

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\activate  # Windows

# Install main task2 requirements
pip install -r task2_neural_network/requirements.txt

# Install task3 scraper requirements
pip install -r task3/Scrapped/requirements.txt
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

Built during a Computer Vision & Machine Learning Internship at **Silvertouch Technologies**  
Python · OpenCV · NumPy · scikit-learn · colour-science · PIL · Shapely

</div>
