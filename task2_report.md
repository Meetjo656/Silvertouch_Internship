# Task 2 — Structural Packaging Intelligence (Geometric ML)

> **Domain:** Geometric Machine Learning / Computer Vision
> **Type:** Multi-phase research pipeline (geometry + classification)
> **Status:** Partially runnable — active research WIP
> **Folder:** `task2_neural_network/`

---

## 1. Overview

Task 2 reconstructs the **structural layout of packaging / carton blueprints** from images. Instead of relying on wavy deep-learning bounding boxes, it extracts straight structural lines, snaps their intersections into a **Planar Straight Line Graph (PSLG)**, polygonizes the result into clean faces, and classifies each face (e.g. PANEL, FLAP, GLUE_TAB, DUST_FLAP) using **Random Forest** classifiers trained on geometric features. The selling point is topological correctness that raw CNNs cannot guarantee.

## 2. Problem Statement

Manufacturing packaging (flaps, tabs, panels) requires strict CAD-like, non-overlapping geometry. A PSLG data structure mathematically blocks overlapping segments (no two lines intersect except at shared vertices), partitioning a carton's layout cleanly — something pixel-wise detectors cannot promise.

## 3. Architecture & Modules (`src/`)

| Subpackage | Notable files | Responsibility |
|---|---|---|
| `core/` | `labels.py` | `StructuralClass` enum: PANEL, FLAP, GLUE_TAB, DUST_FLAP, BACKGROUND, UNKNOWN |
| `geometry/` (17 files) | `edge_detector.py`, `line_detector.py`, `line_regularizer.py`, `pslg_builder.py`, `polygonizer.py`, `gap_bridge_solver.py`, `boundary_completion_engine.py`, `structural_edge_scorer.py`, `structural_connectivity.py`, `silhouette_fusion.py` | Hough line detection → PSLG construction → polygonization |
| `features/` | `geometric.py` | `extract_geometric_features()` — 17-D feature vector |
| `models/` | `rf_classifier.py` | `StructuralMaskRFClassifier` (RandomForest + StandardScaler) |
| `blueprint/` (10 files) | `planar_partition_builder.py`, `fragment_merger.py`, `structural_region_merger.py`, `containment_filter.py`, `tiny_face_filter.py`, `blueprint_validator.py`, `region_graph_builder.py` | Face-level refinement (Stages 4+) |
| `analysis/` (10 files) | `dataset_dashboard.py`, `structural_existence_audit.py`, `outer_boundary_audit.py`, `geometry_failure_audit.py`, `region_overlap_analysis.py` | Audits / evaluations |
| `data/` | `metadata_mgr.py` (MetadataManager, pydantic), `dataset_builder.py` (MaskDatasetBuilder), `sam_pipeline.py` (SAMPipelineIngestor) | Mask loading + metadata |
| `pipeline/` | `geometry_pipeline.py` (run_geometry_pipeline, Stages 0–4), `phase_6_pipeline.py`, `phase_7a_analysis.py`, `phase_7b_lite_pipeline.py` | End-to-end orchestration |
| `training/` | `train.py` (`train_model`), `evaluate.py`, `generate_report.py` | Model training / reporting |
| `tools/` (13 files) | `auto_label_multiclass.py`, `label_gui.py`, `validate_dataset.py`, `audit_sam_masks.py` | Labeling & QA utilities |
| `visualization/` | `face_boundary_renderer.py` | Render faces |
| `utils/` | `logger.py` (`get_logger`) | Logging |

Plus root scripts: `analyze_masks.py` (coverage histograms → `data/interim`), `generate_and_ingest.py` (external SAM segmentation → masks).

## 4. Key Files & Entry Points

- **`src/pipeline/geometry_pipeline.py` → `run_geometry_pipeline()`** — main flow: ArtworkSuppression → Edges → Hough Lines → Regularize → PSLG → Polygonize → PlanarPartition.
- **`generate_and_ingest.py`** — moves loose images to `data/raw_temp`, runs an external `SAMSegmenter` (`task2.segmentation.sam_segmenter`, **not in repo**; needs `D:\...\weights\sam_vit_h.pth`), then `SAMPipelineIngestor.ingest_image_and_masks()` writes `data/raw/masks` + updates `dataset_metadata.json`.
- **`analyze_masks.py` → `analyze()`** — per-mask area / relative_area / cumulative coverage; writes PNGs to `data/interim`.
- **`src/training/train.py`** — features → `GroupShuffleSplit(80/20)` → train → save `structural_component_rf.joblib`.

## 5. Core Techniques

- **Hough lines** — `LineDetector.detect_lines()` uses `cv2.HoughLinesP` (threshold 50, `minLineLength` = 2% image diagonal, `maxLineGap` 10).
- **PSLG** — `PSLGBuilder.build_graph()` extends each line by ~1% image diagonal, unions via shapely `unary_union` (auto-nodes intersections), prunes tiny edges.
- **Polygonize** — `Polygonizer.polygonize_lines()` → shapely `polygonize()` → closed faces → GeoJSON.
- **17 geometric features** — area, relative_area, perimeter, bbox_width, bbox_height, aspect_ratio, solidity, extent, equiv_diameter, circularity, rotated_width/height/aspect_ratio/area, vertex_count (`approxPolyDP`), centroid_x/y.
- **Random Forest** — `StructuralMaskRFClassifier` (`n_estimators` 100/500, `max_depth=None`, `class_weight="balanced"`, StandardScaler). Two artifacts: `baseline_rf.joblib` (binary structural/non-structural) and `structural_component_rf.joblib` (multiclass). Paired scalers + `X_test.npy`/`y_test.npy`.

## 6. Tech Stack / Dependencies

From `requirements.txt`: numpy, pandas, opencv-python, scikit-image (declared, **unused** in source), scikit-learn, pydantic, matplotlib, seaborn, joblib.

> ⚠️ **`shapely` is used extensively** (PSLG / polygonizer / blueprint) but is **missing from `requirements.txt`** — a real install gap.
> ⚠️ Depends on an **external `task2` package** (`SAMSegmenter`) not present in this repo.

## 7. Data & Artifacts

- `data/raw/images` (7), `data/raw/masks` (371); `data/interim/` (186 files: geometry_pipeline, outer_boundary_audit, phase_7b_lite, statistics, topo_reviews); `data/output_segments/` (7 `*_blueprint.jpg`); `data/metadata/dataset_metadata.json` (per-image records: `structural_class`, `is_candidate`, `skip_reason`).
- `data/processed` is **empty** (end-to-end pipeline not fully executed).
- `models/`: `baseline_rf.joblib`, `structural_component_rf.joblib`, paired scalers, `X_test.npy`, `y_test.npy`.

## 8. Maturity / Status

**Partially runnable, research-stage.** `tests/` is **empty** (no unit tests). `generate_and_ingest.py` depends on a Windows-absolute SAM weights path + out-of-repo module. `baseline` model is loaded in `generate_report.py` but its training script is not in `src/`. Much of geometry/blueprint is a 7-phase research pipeline (phases 5–7). Not production-ready; best read as an evolving geometric-ML study.

## 9. How to Run (inferred)

```bash
pip install numpy pandas opencv-python scikit-learn pydantic matplotlib seaborn joblib shapely
# 1. Label masks (tools/auto_label_multiclass.py or label_gui.py) → update metadata
# 2. Geometry-first reconstruction:
python -m src.pipeline.geometry_pipeline
# 3. Face cleanup / merge:
python -m src.pipeline.phase_7b_lite_pipeline
# 4. Train classifier:
python -m src.training.train --n_estimators 500
# 5. Evaluate / report:
python -m src.training.evaluate
python -m src.training.generate_report
# Masks analysis:
python analyze_masks.py
```

**Key gaps to flag:** missing `shapely` in requirements, external SAM dependency, empty `tests/`, and an untracked `baseline` training step.

---

*Report generated from source inspection of `task2_neural_network/`. See parent `README.md` for the portfolio-level summary of this task.*
