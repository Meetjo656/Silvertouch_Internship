# 🧪 Silvertouch Internship — Computer Vision & Machine Learning Pipeline

> **Pharmaceutical Packaging Analysis** using perceptual color science, geometry-first structural layout reconstruction, deep segmentation models, custom data scraping, and **LLM-powered Semantic SQL Generation with Entity Extraction & Context Routing**.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Latest-green)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Transformers](https://img.shields.io/badge/Transformers-4.53%2B-purple)

---

## 📚 Quick Navigation

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Task 1: Text Color Comparison](#task-1-perceptual-text-color-comparison)
- [Task 2: Structural Layout Reconstruction](#task-2-structural-packaging-layout-reconstruction)
- [Task 3: Data Pipeline](#task-3-dailymed-scraper--dataset-curation)
- [Task 4: LLM-Powered Semantic SQL Generation](#task-4-llm-powered-semantic-sql-generation)
- [Setup & Installation](#setup--installation)

---

## 🔭 Overview

This repository contains **computer vision, machine learning, and NLP pipelines** developed during a Silvertouch Technologies internship, focusing on:
1. **Pharmaceutical packaging design and structural analysis** (Tasks 1-3)
2. **Enterprise semantic understanding and SQL generation via LLMs** (Task 4)

### Project Components

| **Component** | **Task 1** | **Task 2** | **Task 3** | **Task 4** |
|---|---|---|---|---|
| **Objective** | Perceptually compare text color between crop images | Reconstruct packaging structural layout | Scrape & curate pharmaceutical packaging datasets | Generate SQL from natural language via semantic routing |
| **Core Technology** | LAB MeanShift + ΔE CIE2000 | PSLG builder + Polygonizer + Random Forest | Multi-threaded Scraper + PIL Lanczos | Transformers + LLM Fine-tuning + FAISS Embeddings |
| **Input** | Two cropped text images | Carton images + SAM segment masks | DailyMed SETIDs / Roboflow annotated dataset | Natural language queries + enterprise semantic catalog |
| **Output** | Comparison dashboard PNG | GeoJSON partitions, boundary overlay PNGs | Deduplicated artwork, resized images, unified COCO annotations | SQL queries + ranked context + semantic entity extraction |
| **Key Metric** | ΔE ≤ 8.0 → MATCH | Coverage %, Overlap %, classification accuracy | Scrape speed, deduplication rate | MRR (Mean Reciprocal Rank), F1 (entity extraction) |

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
└── task4/                                  # LLM-Powered Semantic SQL Generation & Entity Extraction
    ├── requirements.txt                    # Transformers, PEFT, TRL, FastAPI, SQLAlchemy
    ├── data/
    │   ├── raw/                            # Source datasets (CSV: SQL, DDL, docs)
    │   ├── processed/                      # Generated train.jsonl for fine-tuning
    │   ├── embeddings/                     # Cached FAISS embeddings & indices
    │   ├── semantic_catalog/               # Entity definitions & metadata
    │   ├── benchmark/                      # Test queries & evaluation sets
    │   └── enterprise_semantic_catalog.json # Master semantic ontology
    ├── outputs/                            # Model outputs & evaluation reports
    ├── scripts/
    │   └── analyze_dataset.py              # Statistics on instruction/SQL lengths
    ├── src/
    │   ├── context/
    │   │   └── user_context.py             # User role, dept, state, district context
    │   ├── data_pipeline/
    │   │   ├── build_dataset.py            # Joins DDL/docs/SQL into JSONL
    │   │   ├── load_schema.py              # Loads database schemas
    │   │   └── show_head.py                # Displays first rows of raw data
    │   ├── ingestion/
    │   │   ├── build_corpus.py             # Builds semantic corpus
    │   │   └── profiler.py                 # Profiles entity frequencies & patterns
    │   ├── retrieval/
    │   │   ├── base.py                     # Abstract Retriever base class
    │   │   ├── ddl_retriever.py            # Retrieves relevant DDL/schema definitions
    │   │   ├── doc_retriever.py            # Retrieves documentation snippets
    │   │   ├── sql_retriever.py            # Retrieves similar SQL examples
    │   │   ├── embedding_retriever.py      # FAISS-based semantic retrieval
    │   │   ├── context_ranker.py           # Ranks & combines multi-source context
    │   │   └── assembler.py                # Constructs final prompt/context
    │   ├── routing/
    │   │   ├── router.py                   # Base routing logic
    │   │   ├── semantic_router.py          # Semantic intent classification
    │   │   ├── retrieval_router.py         # Routes to appropriate retriever
    │   │   ├── extractor.py                # Entity extraction from queries
    │   │   └── build_module_embeddings.py  # Pre-computes module/intent embeddings
    │   ├── evaluation/
    │   │   ├── evaluator.py                # Core evaluation metrics
    │   │   ├── evaluate_routers.py         # Benchmarks routing accuracy
    │   │   ├── evaluate_context_ranking.py # Evaluates retriever ranking
    │   │   ├── quality_analyzer.py         # Analyzes SQL/output quality
    │   │   ├── enrich_benchmark.py         # Augments benchmark queries
    │   │   └── generate_benchmark.py       # Creates new benchmark sets
    │   └── utils/
    │       ├── file_ops.py                 # File I/O helpers
    │       └── generate_reports.py         # Generates evaluation reports
    └── tests/
        ├── test_entity_extractor.py        # Entity extraction tests
        ├── test_pipeline.py                # End-to-end pipeline tests
        ├── test_retrievers.py              # Retriever component tests
        └── test_router.py                  # Routing logic tests
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

## 🚀 Task 4 — LLM-Powered Semantic SQL Generation

**Objective:** Build an **enterprise semantic understanding system** that translates natural language queries into SQL using fine-tuned LLMs, semantic routing, multi-source context retrieval, and entity extraction.

### Architecture Overview

```
User Query + User Context
    │
    ├─→ Semantic Router
    │       ├─→ Semantic Intent Classification (Embedding-based)
    │       ├─→ Entity Extraction (Extract entities: State, District, etc.)
    │       └─→ Route to appropriate retrieval pipeline
    │
    ├─→ Multi-Source Context Retrieval
    │       ├─→ DDL Retriever (Schema definitions)
    │       ├─→ Doc Retriever (Documentation snippets)
    │       ├─→ SQL Retriever (Similar SQL examples)
    │       └─→ Embedding Retriever (FAISS semantic search)
    │
    ├─→ Context Ranking & Assembly
    │       ├─→ Multi-source combination
    │       ├─→ Relevance ranking (BM25 + embeddings)
    │       └─→ Final prompt construction
    │
    ├─→ LLM Inference
    │       ├─→ Fine-tuned Model (PEFT/LoRA)
    │       └─→ Generate SQL with explanation
    │
    └─→ Output Assembly
        ├─→ SQL query
        ├─→ Confidence score
        ├─→ Extracted entities
        └─→ Retrieved context metadata
```

### Core Modules

#### `context/ — User Context Management`

| File | Class | Purpose |
|---|---|---|
| `user_context.py` | `UserContext` | Dataclass storing user role, department, state, district, city for personalized routing |

**Example:**
```python
context = UserContext(
    role="analyst",
    department="agricultural_extension",
    state="Maharashtra",
    district="Pune",
    city="Pune"
)
```

#### `data_pipeline/ — Dataset Building & Preparation`

| File | Function | Purpose |
|---|---|---|
| `build_dataset.py` | Joins raw CSVs | Combines DDL, documentation, and SQL examples into JSONL format for fine-tuning |
| `show_head.py` | Display data samples | Previews raw SQL, DDL, and documentation datasets |
| `load_schema.py` | (Empty placeholder) | Future: Load and parse database schema objects |

**Dataset Format (JSONL):**
```json
{
  "instruction": "Schema:\n{DDL}\n\nDocumentation:\n{DOCS}\n\nQuestion:\n{PROMPT}",
  "response": "{SQL}"
}
```

#### `ingestion/ — Corpus & Entity Profiling`

| File | Class/Function | Purpose |
|---|---|---|
| `profiler.py` | `EntityProfiler` | Analyzes entity frequencies, distributions, and semantic relationships in corpus |
| `build_corpus.py` | (Empty placeholder) | Future: Constructs semantic corpus from heterogeneous sources |

**EntityProfiler Output:**
```json
{
  "entity_type": "State",
  "frequency": 1234,
  "examples": ["Maharashtra", "Karnataka", ...],
  "related_entities": {"District": [...]}
}
```

#### `retrieval/ — Multi-Source Context Retrieval`

| File | Class | Purpose |
|---|---|---|
| `base.py` | `BaseRetriever` | Abstract base class defining retriever interface |
| `ddl_retriever.py` | `DDLRetriever` | Retrieves relevant table/column DDL definitions |
| `doc_retriever.py` | `DocRetriever` | Retrieves documentation for tables & columns |
| `sql_retriever.py` | `SQLRetriever` | Retrieves similar SQL examples (lexical + semantic) |
| `embedding_retriever.py` | `EmbeddingRetriever` | FAISS-based semantic retrieval using pre-computed embeddings |
| `context_ranker.py` | `ContextRanker` | Combines & ranks results from multiple retrievers |
| `assembler.py` | `PromptAssembler` | Constructs final prompt with ranked context |

**BaseRetriever Interface:**
```python
class BaseRetriever:
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Returns list of retrieved contexts with scores."""
        pass
```

#### `routing/ — Semantic Intent & Entity Extraction`

| File | Class/Function | Purpose |
|---|---|---|
| `router.py` | `BaseRouter` | Base routing logic |
| `semantic_router.py` | `SemanticRouter` | Classifies query intent using embeddings (e.g., "Statistics", "Hierarchy", "Drill-down") |
| `retrieval_router.py` | `RetrievalRouter` | Routes to appropriate retriever based on intent |
| `extractor.py` | `EntityExtractor` | Extracts semantic entities (State, District, etc.) from query using regex/ML |
| `build_module_embeddings.py` | Pre-computes embeddings | Builds FAISS indices for module/intent classification |

**SemanticRouter Output:**
```json
{
  "intent": "statistics",
  "entities": {
    "State": "Maharashtra",
    "Sector": "Dairy"
  },
  "confidence": 0.94
}
```

#### `evaluation/ — Benchmarking & Quality Analysis`

| File | Class/Function | Purpose |
|---|---|---|
| `evaluator.py` | `Evaluator` | Core metrics: Exact Match (EM), F1 (partial match), BLEU |
| `evaluate_routers.py` | Router accuracy evaluation | Benchmarks semantic routing & entity extraction F1 |
| `evaluate_context_ranking.py` | Context relevance evaluation | Measures retriever MRR (Mean Reciprocal Rank), NDCG |
| `quality_analyzer.py` | `QualityAnalyzer` | Detects SQL syntax errors, runtime issues, schema violations |
| `generate_benchmark.py` | Creates benchmark sets | Generates diverse test queries from raw SQL dataset |
| `enrich_benchmark.py` | Augments benchmarks | Paraphrases & variations on existing benchmark queries |

**Evaluation Metrics:**
```
EM (Exact Match):    Correct SQL matches exactly
F1 (Partial Match):  Token-level overlap
BLEU:                N-gram similarity
MRR (Retrieval):     Reciprocal rank of first relevant result
NDCG (Retrieval):    Normalized Discounted Cumulative Gain
Entity F1:           Precision & Recall of extracted entities
```

#### `utils/ — Utilities & Reporting`

| File | Function | Purpose |
|---|---|---|
| `file_ops.py` | File I/O helpers | Load/save JSONL, JSON, CSV with error handling |
| `generate_reports.py` | Report generation | Creates evaluation reports, confusion matrices, performance summaries |

### Semantic Catalog

**`enterprise_semantic_catalog.json`** defines the enterprise's semantic ontology:

```json
[
  {
    "concept": "State",
    "concept_type": "administrative_unit",
    "aliases": [],
    "definition": "The primary administrative division of India.",
    "modules": ["GP_COVERAGE", "COOPERATIVE", "RANKINGS"],
    "related_tables": ["chatbot_districts_blocks_gp_villages"],
    "related_columns": ["state_name", "state_code"],
    "parent_concepts": ["India"],
    "child_concepts": ["District"],
    "confidence": 0.99
  },
  ...
]
```

**Usage:**
- Router uses catalog to classify intents → "State" entity → find module
- Context retriever uses catalog to find related tables/columns
- Evaluator validates extracted entities against catalog definitions

### Requirements

**`task4/requirements.txt`:**
```
transformers>=4.53.0
datasets>=3.6.0
accelerate>=1.8.0
evaluate>=0.4.3

peft>=0.16.0
trl>=0.19.0
bitsandbytes>=0.46.0

sentencepiece>=0.2.0
tiktoken>=0.9.0

pandas>=2.3.0
numpy>=2.3.0
pyarrow>=20.0.0

pyyaml>=6.0.2
omegaconf>=2.3.0

tensorboard>=2.19.0
wandb>=0.21.0

tqdm>=4.67.1
rich>=14.0.0

scikit-learn>=1.7.0
sqlparse>=0.5.3
sqlalchemy>=2.0.41

fastapi>=0.115.13
uvicorn>=0.34.3
```

### Data Structure

```
task4/data/
├── raw/
│   └── banks_sample_csv/
│       ├── pgviews_banks_sql_train.csv      # Natural language + SQL pairs
│       ├── pgviews_banks_ddl_train.csv      # Table/column definitions
│       └── banks_documentation_train.csv    # Documentation strings
├── processed/
│   └── train.jsonl                          # Ready for fine-tuning
├── embeddings/
│   ├── faiss_index.bin                      # FAISS vector index
│   └── embeddings.npy                       # Cached embeddings
├── semantic_catalog/
│   ├── entities.json                        # Entity type definitions
│   └── modules.json                         # Module/intent definitions
└── benchmark/
    ├── test_queries.json                    # Evaluation queries
    └── ground_truth.json                    # Expected SQL outputs
```

### Running Task 4

```bash
cd task4

# 1. Build training dataset
python -m src.data_pipeline.build_dataset

# 2. Analyze dataset statistics
python scripts/analyze_dataset.py

# 3. Build embeddings & FAISS index
python -m src.routing.build_module_embeddings

# 4. Fine-tune model (PEFT/LoRA)
python -m src.training.train \
  --model_name "meta-llama/Llama-2-7b" \
  --learning_rate 1e-4 \
  --num_epochs 3

# 5. Evaluate routers & retrievers
python -m src.evaluation.evaluate_routers
python -m src.evaluation.evaluate_context_ranking

# 6. Generate & enrich benchmark
python -m src.evaluation.generate_benchmark
python -m src.evaluation.enrich_benchmark

# 7. Run full inference pipeline
python -m src.pipeline.inference --query "Show me dairy cooperatives in Maharashtra"
```

### API Usage Example

```python
from src.routing.semantic_router import SemanticRouter
from src.retrieval.context_ranker import ContextRanker
from src.context.user_context import UserContext

# Initialize components
router = SemanticRouter()
ranker = ContextRanker()

# User context & query
context = UserContext(role="analyst", department="cooperative", 
                      state="Maharashtra", district="Pune")
query = "How many primary agricultural credit societies operate in Gram Panchayats?"

# Route & extract entities
routing_result = router.route(query, context)
print(f"Intent: {routing_result['intent']}")
print(f"Entities: {routing_result['entities']}")

# Retrieve context from multiple sources
ddl_results = retriever.retrieve_ddl(routing_result['entities'])
doc_results = retriever.retrieve_docs(routing_result['entities'])
sql_results = retriever.retrieve_similar_sql(query)

# Rank & assemble
ranked_context = ranker.rank([ddl_results, doc_results, sql_results])
prompt = assembler.assemble(query, ranked_context, context)

# LLM inference
generated_sql = model.generate(prompt)
```

### Example Outputs

**Query:** "Show state-wise dairy cooperative statistics for cooperatives with members > 1000"

**Routing Output:**
```json
{
  "intent": "statistics",
  "entities": {
    "State": null,
    "Sector": "Dairy"
  },
  "modules": ["RANKINGS", "FEDERATION"],
  "confidence": 0.97
}
```

**Retrieved Context:**
```json
{
  "ddl": [
    "CREATE TABLE chatbot_state_district_federation_private (state_name VARCHAR, sector_of_operation_name VARCHAR, ...)",
    "CREATE TABLE chatbot_cooperative_registrations_master (sector_of_operation_name VARCHAR, members_count INT, ...)"
  ],
  "docs": [
    "Dairy sector: Cooperatives engaged in milk procurement, processing, and marketing.",
    "Federation table contains aggregated state-district level statistics."
  ],
  "similar_sql": [
    "SELECT state_name, COUNT(*) FROM chatbot_state_district_federation_private WHERE sector='Dairy' GROUP BY state_name;"
  ]
}
```

**Generated SQL:**
```sql
SELECT 
  state_name,
  COUNT(*) as cooperative_count,
  AVG(members_count) as avg_members
FROM chatbot_state_district_federation_private sdrf
JOIN chatbot_cooperative_registrations_master crm 
  ON sdrf.federation_code = crm.federation_code
WHERE sdrf.sector_of_operation_name = 'Dairy'
  AND crm.members_count > 1000
GROUP BY state_name
ORDER BY cooperative_count DESC;
```

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
  - **For Task 4:** `transformers`, `peft`, `trl`, `fastapi`, `sqlalchemy` — LLM & API support

### Virtual Environment Setup

```bash
# Clone repository
git clone <repo-url>
cd Silvertouch_Internship

# Create virtual environment
python -m venv .venv
source .venv/bin/activate           # macOS/Linux
# .venv\Scripts\activate             # Windows

# Install Task 1 dependencies (base)
pip install opencv-python numpy scikit-learn colour-science scikit-image pandas shapely matplotlib seaborn joblib imagehash

# Install Task 2 dependencies
pip install -r task2_neural_network/requirements.txt

# Install Task 3 dependencies
pip install -r task3/Scrapped/requirements.txt

# Install Task 4 dependencies
pip install -r task4/requirements.txt
```

### Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `SAM_CHECKPOINT` | `task2/weights/sam_vit_h.pth` | Path to SAM ViT-H weights |
| `HUGGINGFACE_TOKEN` | (required for Task 4) | HuggingFace API token for model downloads |
| `FAISS_OMP_NUM_THREADS` | `4` | Threads for FAISS operations |

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

### Task 4 LLM Fine-tuning

| Parameter | Default | Effect |
|---|---|---|
| `learning_rate` | `1e-4` | Lower → more stable; higher → faster convergence |
| `num_epochs` | `3` | More epochs → better fit but risk of overfitting |
| `lora_rank` | `8` | Higher → more expressive; lower → smaller memory |
| `lora_alpha` | `16` | Controls update magnitude |
| `retrieval_top_k` | `5` | Number of context examples to retrieve |

### Task 4 Router Thresholds

| Parameter | Default | Effect |
|---|---|---|
| `intent_confidence_threshold` | `0.85` | Lower → more retrievals; higher → less fallback |
| `entity_extraction_confidence` | `0.8` | Lower → more entities extracted |
| `context_ranking_top_k` | `5` | Results to pass to LLM |

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

### Task 3

| File | Contents |
|---|---|
| `task3/train/` | Training split images & annotations |
| `task3/test/` | Test split images & annotations |
| `task3/valid/` | Validation split images & annotations |
| `task3/Scrapped/output/images/` | Downloaded DailyMed artwork |
| `task3/Scrapped/output/metadata/images.csv` | Image metadata (URL, size, hash) |
| `task3/Scrapped/output/metadata/duplicates.csv` | Duplicate image tracking |

### Task 4

| File | Contents |
|---|---|
| `task4/data/processed/train.jsonl` | Fine-tuning dataset |
| `task4/data/embeddings/faiss_index.bin` | FAISS vector index |
| `task4/outputs/eval_report.json` | Evaluation metrics (EM, F1, BLEU, MRR, NDCG) |
| `task4/outputs/benchmark_results.json` | Per-query routing & retrieval results |
| `task4/outputs/quality_report.json` | SQL syntax & semantic error analysis |
| `task4/models/lora_adapter.safetensors` | PEFT LoRA weights |

---

## 🔗 Useful Links

- 📄 [Roboflow Dataset Info](task3/README.roboflow.txt)
- 📊 [Dataset Statistics](task3/README.dataset.txt)
- 🧬 [DailyMed Scraper README](task3/Scrapped/README.md)
- 📚 [Semantic Catalog](task4/data/enterprise_semantic_catalog.json)

---

<div align="center">

**Built during a Computer Vision, Machine Learning & NLP Internship at Silvertouch Technologies**

🐍 Python · 🎥 OpenCV · 🔢 NumPy · 🤖 scikit-learn · 🌈 colour-science · 🖼️ PIL · 📐 Shapely · 🦙 Transformers · ⚡ PEFT

[Repository](https://github.com/Meetjo656/Silvertouch_Internship) | [Issues](https://github.com/Meetjo656/Silvertouch_Internship/issues)

</div>
