# 🧪 Silvertouch Internship — Computer Vision & Machine Learning Pipeline

> **Pharmaceutical Packaging Analysis** using perceptual color science, geometry-first structural layout reconstruction, deep segmentation models, and custom data scraping.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Latest-green)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)

---

## 📚 Quick Navigation

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Task 1: Text Color Comparison](#task-1-perceptual-text-color-comparison)
- [Task 2: Structural Layout Reconstruction](#task-2-structural-packaging-layout-reconstruction)
- [Task 3: Data Pipeline](#task-3-dailymed-scraper--dataset-curation)
- [Task 4: Future Work](#task-4-future-workspace)
- [Setup & Installation](#setup--installation)

---

## 🔭 Overview

This repository contains **computer vision and machine learning pipelines** developed during a Silvertouch Technologies internship, focusing on **pharmaceutical packaging design and structural analysis**.

### Project Components

| **Component** | **Task 1** | **Task 2** | **Task 3** |
|---|---|---|---|
| **Objective** | Perceptually compare text color between crop images | Reconstruct packaging structural layout (panels, flaps, glue tabs) | Scrape & curate pharmaceutical packaging datasets |
| **Core Technology** | LAB MeanShift clustering + ΔE CIE2000 | PSLG builder + Polygonizer + Random Forest | Multi-threaded Scraper + PIL Lanczos + COCO mapping |
| **Input** | Two cropped text images | Carton images + SAM segment masks | DailyMed SETIDs / Roboflow annotated dataset |
| **Output** | Comparison dashboard PNG | GeoJSON partitions, boundary overlay PNGs, trained classifier | Deduplicated artwork, resized images, unified COCO annotations |
| **Key Metric** | ΔE ≤ 8.0 → MATCH | Coverage %, Overlap %, classification accuracy | Scrape speed, deduplication rate, resolution consistency |

---

## 📁 Repository Structure

```
Silvertouch_Internship/
│
├── .gitignore
├── README.md
│
├── src/                                    # Shared Utilities
│   └── visualization/
│       └── face_boundary_renderer.py       # Renders recovered polygon faces as boundaries
│
├── task1/                                  # Text Color Comparison Pipeline
│   ├── main.py                             # Entry point
│   ├── config.py                           # Configuration & thresholds
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
├── task2_neural_network/                   # Structural Layout Reconstruction & ML Pipeline
│   ├── generate_and_ingest.py              # Ingests images and SAM masks
│   ├── analyze_masks.py                    # Computes area histograms & coverage stats
│   ├── requirements.txt
│   ├── config/                             # Configuration files
│   ├── data/
│   │   ├── raw/                            # Source carton images
│   │   ├── raw_temp/                       # Temporary processing storage
│   │   ├── interim/                        # Intermediate outputs (GeoJSONs, CSV stats)
│   │   ├── metadata/                       # JSON dataset metadata
│   │   ├── processed/                      # Final processed datasets
│   │   └── output_segments/
│   ├── models/                             # Trained models (.joblib outputs)
│   ├── notebooks/                          # Jupyter Notebooks
│   ├── tests/                              # Unit tests
│   └── src/
│       ├── analysis/                       # Audits (Boundary, ambiguity, overlap, topology)
│       ├── blueprint/                      # Planar partitioning, validation, filters
│       ├── core/                           # Shared data structures & enums
│       ├── data/                           # Dataset building & SAM pipeline ingestion
│       ├── features/                       # Geometric feature extraction
│       ├── geometry/                       # Line detection, PSLG, polygonizer
│       ├── models/                         # Baseline models (Random Forest)
│       ├── pipeline/                       # Orchestration
│       ├── tools/                          # Audits & labeling tools
│       ├── training/                       # Model training scripts
│       └── utils/                          # Shared helpers & loggers
│
├── task3/                                  # DailyMed Scraper & Dataset Curation
│   ├── README.dataset.txt
│   ├── README.roboflow.txt
│   ├── combine_eval.py                     # Combines test/validation splits
│   ├── migrate_test.py                     # Maps test set annotations
│   ├── migrate_valid.py                    # Maps validation set annotations
│   ├── resize_dataset.py                   # Standardizes image sizes (max 1000px)
│   ├── train/                              # Training split
│   ├── test/                               # Testing split
│   ├── valid/                              # Validation split
│   └── Scrapped/                           # DailyMed artwork scraper
│       ├── README.md
│       ├── requirements.txt
│       ├── output/
│       │   ├── images/                     # Downloaded packaging artwork
│       │   ├── metadata/                   # Metadata (images.csv, duplicates.csv)
│       │   └── logs/                       # Detailed logging
│       └── src/
│           ├── main.py
│           ├── crawler.py
│           ├── downloader.py
│           ├── filters.py
│           ├── parser.py
│           └── utils.py
│
└── task4/                                  # Future workspace
```

---

## 🎨 Task 1 — Perceptual Text-Color Comparison

**Objective:** Determine whether the **text color** on two pharmaceutical carton crops is **perceptually identical**, matching human visual perception rather than raw RGB values.

### Pipeline Architecture

```
reference.png                target.png
    │                            │
    ├─→ load_image()             │
    │       │                    ├─→ load_image()
    │       ▼                    │       │
    │   extract_foreground()      │       ▼
    │       │                    │   extract_foreground()
    │       ▼                    │       │
    │   MeanShiftExtractor       │       ▼
    │   (LAB space)              │   MeanShiftExtractor
    │       │                    │   (LAB space)
    │       ▼                    │       │
    │   get_dominant_color()     │       ▼
    │       │                    │   get_dominant_color()
    │       ▼                    │       │
    │   get_color_family()       │       ▼
    │       │                    │   get_color_family()
    └────→─┴────────────┬────────┴──────→─┘
                        │
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

### Module Breakdown

#### `preprocessing/`

| File | Function | Description |
|---|---|---|
| `load_image.py` | `load_image()` | Read image from disk, validate shape |
| `foreground_extraction.py` | `extract_foreground()` | Build foreground mask by comparing pixel ΔE against estimated background |
| `estimate_background.py` | `estimate_background()` | Sample image corners to derive representative background color in LAB |

**Output of `extract_foreground()`:**
```python
{
    "mask": np.ndarray,                    # binary foreground mask
    "lab": np.ndarray,                     # LAB image (foreground only)
    "delta_from_bg_values": np.ndarray     # per-pixel ΔE from background
}
```

#### `clustering/`

| File | Class | Description |
|---|---|---|
| `meanshift_cluster.py` | `MeanShiftExtractor` | Runs scikit-learn MeanShift on LAB foreground pixels to find density peaks |

MeanShift clusters without requiring a fixed cluster count — peaks represent true dominant ink colors.
- **Configuration:** `MEANSHIFT_QUANTILE = 0.2` (controls bandwidth)

#### `color_processing/`

| File | Function/Class | Description |
|---|---|---|
| `dominant_color.py` | `get_dominant_color()` | Select largest cluster centroid as ink color |
| `color_family.py` | `get_color_family()`, `get_meta_family()` | Map RGB → named color family (e.g., "crimson") → meta family (e.g., "red") |
| `deltae_compare.py` | `CompareColors` | Compute ΔE CIE2000 between two LAB colors; threshold-based MATCH decision |
| `rgb_hex.py` | `rgb_to_hex()` | Format numpy RGB array as `#RRGGBB` |

**Match Decision Logic:**
```python
ΔE ≤ 8.0   →  "MATCH"
ΔE > 8.0   →  "NO MATCH"
```

#### `visualization/`

| File | Function | Description |
|---|---|---|
| `dashboard.py` | `generate_comparison_dashboard()` | Render multi-panel PNG with images, masks, clusters, swatches, metrics |

### Configuration

```python
# task1/config.py

REFERENCE_IMAGE           = "data/reference.png"
TARGET_IMAGE              = "data/target.png"
OUTPUT_DASHBOARD          = "outputs/comparison_dashboard.png"

DELTA_E_MATCH_THRESHOLD   = 8.0      # ΔE above this → NO MATCH
FOREGROUND_DELTA_THRESHOLD = 10      # Min ΔE from BG to count as foreground
MEANSHIFT_QUANTILE        = 0.2      # Bandwidth control for MeanShift
SAVE_DEBUG_IMAGES         = False    # Toggle intermediate image saving
```

### Running Task 1

```bash
cd task1
python main.py
```

**Expected Output:**
```
Stage 1: Initializing Perceptual Text-Color Semantic Comparison Pipeline...
Stage 2: Loading cropped text images...
Stage 3: Extracting foreground text masks...
Stage 4: Performing density-based MeanShift clustering...
Stage 5: Selecting semantic dominant color clusters...
Stage 6: Mapping dominant colors to semantic families...
Stage 7: Generating high-fidelity Comparison Dashboard...
Success: Comparative pipeline executed flawlessly.
```

---

## 🏭 Task 2 — Structural Packaging Layout Reconstruction

**Objective:** Reconstruct structural panels, dust flaps, and glue tabs of pharmaceutical cartons using a **geometry-first planar partition pipeline** combined with machine learning classification.

### Architecture & Pipelines

#### Pipeline 1: Geometry-First Partition

```
Original Image
    │
    ├─→ Stage 0: Artwork Suppression (Bilateral + Opening)
    │       │
    │       ▼
    ├─→ Stage 1: Edge Recovery (Canny + Sobel)
    │       │
    │       ▼
    ├─→ Stage 2: Line Detection & Regularization (Hough)
    │       │
    │       ▼
    ├─→ Stage 3: PSLG Construction & Polygonization
    │       │
    │       ▼
    ├─→ Stage 4: Planar Partition Building
    │       │
    │       ▼
    Outputs: GeoJSON Face Boundaries & _blueprint_final.png
```

#### Pipeline 2: Cleanup & Auditing (Phase 7B-Lite)

Performs topological filtering and structural consolidation:
- **FaceCleanupAudit**: Verifies polygon coordinates against image dimensions
- **ContainmentFilter**: Eliminates nested/duplicate containment loops
- **TinyFaceFilter**: Rejects micro-polygons (noise from labels/text)
- **StructuralRegionMerger**: Consolidates overlapping shapes
- **StructuralConnectivityValidator**: Confirms final boundary connectivity

### Geometric Feature Extraction

`extract_geometric_features()` extracts **17-dimensional geometric profile** from binary masks:

**Basic Properties:**
- `area`, `relative_area`, `perimeter`

**Bounding Attributes:**
- `bbox_width`, `bbox_height`, `aspect_ratio`

**Topological Complexity:**
- `solidity` (contour area vs convex hull)
- `extent` (contour area vs bounding box)
- `vertex_count` (Douglas-Peucker simplification)

**Circular Properties:**
- `equiv_diameter`, `circularity`

**Rotation-Invariant Properties:**
- `rotated_width`, `rotated_height`, `rotated_aspect_ratio`
- `rotated_min_area_rect`

**Spatial Centroids:**
- `centroid_x`, `centroid_y`

### Classification Model

**`StructuralMaskRFClassifier`** wraps scikit-learn `RandomForestClassifier`:

**Leakage Prevention:** Uses `GroupShuffleSplit` (80/20 split by image ID) to ensure masks from same carton never mix between training/validation.

**Structural Labels** (from `labels.py`):
- `PANEL`: Principal structural faces (Front, Back, Side, Top, Bottom)
- `FLAP`: Closing dust flaps
- `GLUE_TAB`: Connection flaps
- `DUST_FLAP`: Dust flaps
- `BACKGROUND` / `UNKNOWN`

### Module Breakdown

#### `pipeline/ & blueprint/`

| File | Class | Role |
|---|---|---|
| `geometry_pipeline.py` | `run_geometry_pipeline()` | Full line detection & planar polygonizer |
| `phase_7b_lite_pipeline.py` | `run_phase_7b_lite_pipeline()` | Consolidation, filtering, validation |
| `containment_filter.py` | `ContainmentFilter` | Filters duplicate nested polygons |
| `tiny_face_filter.py` | `TinyFaceFilter` | Removes micro-polygons |
| `structural_region_merger.py` | `StructuralRegionMerger` | Merges fragmented structures |

#### `geometry/`

| File | Class | Role |
|---|---|---|
| `artwork_suppressor.py` | `ArtworkSuppressor` | Suppresses graphics & text |
| `edge_detector.py` | `EdgeDetector` | Extracts carton edges |
| `line_detector.py` | `LineDetector` | Hough line transforms |
| `line_regularizer.py` | `LineRegularizer` | Groups, snaps, straightens lines |
| `pslg_builder.py` | `PSLGBuilder` | Generates planar straight-line graphs |
| `polygonizer.py` | `Polygonizer` | Extracts closed polygons |

#### `features/ & models/ & training/`

| File | Class/Function | Role |
|---|---|---|
| `geometric.py` | `extract_geometric_features()` | Calculates 17 geometric features |
| `rf_classifier.py` | `StructuralMaskRFClassifier` | scikit-learn RandomForest wrapper |
| `train.py` | `train_model()` | Trains classifier & dumps model |
| `analyze_masks.py` | `analyze()` | Dataset-wide coverage analysis |
| `generate_and_ingest.py` | `main()` | Auto-generates SAM masks |

### Running Task 2

```bash
cd task2_neural_network

# 1. Analyze masks (creates histograms & coverage logs)
python analyze_masks.py

# 2. Train Random Forest classifier
python -m src.training.train --project_root . --n_estimators 500

# 3. Run geometry-first PSLG polygonization
python -m src.pipeline.geometry_pipeline

# 4. Clean faces & render blueprints
python -m src.pipeline.phase_7b_lite_pipeline
```

---

## 📥 Task 3 — DailyMed Scraper & Dataset Curation

**Objective:** Collect pharmaceutical packaging graphics from DailyMed and standardize annotated image datasets for downstream segmentation.

### DailyMed Artwork Scraper

Located in `task3/Scrapped/`, this pipeline crawls and downloads packaging artwork images.

**Key Features:**
- ⚡ **Robust Multi-threading**: Rapid parallel image downloads
- 🔍 **Perceptual Image Hashing**: Deduplication using `imagehash` (phash)
- 🚫 **Logo Filtering**: Rejects tiny files, aspect-ratio outliers, logos, UI graphics
- ⏸️ **Session Resumption**: Tracks downloaded SETIDs for seamless restart

### Roboflow Dataset Curation

Processes and maps Roboflow annotations (COCO format) into standard structural partitions.

#### Resolution Normalization

`resize_dataset.py` uses PIL's Lanczos filter to downsample images to maximum **1000px**, reducing memory footprints.

#### COCO Annotation Mapping

Maps raw category IDs into unified labels:

| ID | Label |
|---|---|
| 1 | Artwork |
| 2 | not_artwork |
| 3 | Front_Panel |
| 4 | Side_Panel |
| 5 | Back_Panel |
| 6 | Extra_Areas |
| 7 | Bottom_Part |
| 8 | Top_Panel |

**Scripts:**
- `migrate_test.py`: Maps test set annotations
- `migrate_valid.py`: Maps validation set annotations
- `combine_eval.py`: Merges test/validation into `eval_combined`

### Module Breakdown

#### `Scrapped/src/` (DailyMed Scraper)

| File | Module | Role |
|---|---|---|
| `main.py` | Entry point | Configures loops, reads SETIDs, drives downloads |
| `crawler.py` | Crawler | Crawls product XML, finds image URLs |
| `downloader.py` | Downloader | Multi-threaded requests & disk I/O |
| `filters.py` | Filter | Rejects thumbnails & non-artwork |
| `parser.py` | Parser | Extracts metadata from HTML/XML |
| `utils.py` | Utilities | Perceptual hashing & deduplication |

#### Dataset Curation Scripts

| File | Purpose |
|---|---|
| `resize_dataset.py` | Downsample images to max 1000px |
| `combine_eval.py` | Merge test/validation splits |
| `migrate_test.py` / `migrate_valid.py` | Category mapping per split |

### Running Task 3

```bash
cd task3

# 1. Run DailyMed scraper
python Scrapped/src/main.py

# 2. Downsample dataset
python resize_dataset.py

# 3. Map COCO annotations
python migrate_test.py
python migrate_valid.py

# 4. Combine splits for evaluation
python combine_eval.py
```

---

## 🔭 Task 4 — Future Workspace

The `task4/` directory is a clean workspace initialized for future computer vision research, modeling pipelines, or dataset curation scripts.

---

## ⚙️ Setup & Installation

### Requirements

- **Python:** 3.11+
- **Core Dependencies:**
  - `opencv-python` — Image processing
  - `numpy` — Numerical computing
  - `scikit-learn` — Machine learning
  - `colour-science` — Perceptual color operations
  - `scikit-image` — Advanced image operations
  - `pandas` — Data manipulation
  - `shapely` — Geometric operations
  - `matplotlib`, `seaborn` — Visualization
  - `joblib` — Model serialization
  - `ImageHash` — Perceptual image hashing

### Virtual Environment Setup

```bash
# Clone repository
git clone <repo-url>
cd Silvertouch_Internship

# Create virtual environment
python -m venv .venv
source .venv/bin/activate           # macOS/Linux
# .venv\Scripts\activate             # Windows

# Install Task 2 dependencies
pip install -r task2_neural_network/requirements.txt

# Install Task 3 dependencies
pip install -r task3/Scrapped/requirements.txt
```

### Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `SAM_CHECKPOINT` | `task2/weights/sam_vit_h.pth` | Path to SAM ViT-H weights |

---

## 🎛️ Thresholds & Tuning

### Task 1 Parameters

| Parameter | Default | Effect |
|---|---|---|
| `DELTA_E_MATCH_THRESHOLD` | `8.0` | Raise → more lenient color matching |
| `FOREGROUND_DELTA_THRESHOLD` | `10` | Raise → stricter foreground extraction |
| `MEANSHIFT_QUANTILE` | `0.2` | Lower → finer clusters; raise → fewer, broader |

### Task 2 Text/Structure Classifier

| Feature | Text Threshold | Structure Threshold | Rationale |
|---|---|---|---|
| `mask_area` | `< 3,000 px` | `> 5,000 px` | Text is small |
| `fill_ratio` | `< 0.45` | `> 0.50` | Text is sparse in bbox |

### Task 2 MaskReconstructor

| Parameter | Default | Effect |
|---|---|---|
| `CLOSE_KERNEL_SIZE` | `(5, 5)` | Larger → bridges wider gaps |
| `MIN_COMPONENT_AREA` | `100 px` | Raise → more aggressive noise removal |

---

## 📦 Output Artifacts

### Task 1

| File | Contents |
|---|---|
| `outputs/comparison_dashboard.png` | Side-by-side comparison: images, masks, clusters, swatches, ΔE, status |

### Task 2 (per image in `data/output/<image_name>/`)

| File | Contents |
|---|---|
| `overlay.png` | Colored region overlay |
| `segment_map.png` | False-color segment map |
| `segments.json` | Labeled segments with bounding boxes |
| `report.json` | Structural evaluation report |
| `sam_overlay.png` | SAM colored mask overlay |
| `sam_report.json` | Full SAM evaluation |
| `sam_summary.json` | Compact SAM summary |
| `debug/` | Intermediate frames from both pipelines |

---

## 🔗 Useful Links

- 📄 [Roboflow Dataset Info](task3/README.roboflow.txt)
- 📊 [Dataset Statistics](task3/README.dataset.txt)
- 🧬 [DailyMed Scraper README](task3/Scrapped/README.md)

---

<div align="center">

**Built during a Computer Vision & Machine Learning Internship at Silvertouch Technologies**

🐍 Python · 🎥 OpenCV · 🔢 NumPy · 🤖 scikit-learn · 🌈 colour-science · 🖼️ PIL · 📐 Shapely

[Repository](https://github.com/Meetjo656/Silvertouch_Internship) | [Issues](https://github.com/Meetjo656/Silvertouch_Internship/issues)

</div>
