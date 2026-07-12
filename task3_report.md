# Task 3 — Production YOLOv8 / RF-DETR Packaging & Panel Detection Pipeline

> **Domain:** Computer Vision / Data Engineering / MLOps
> **Type:** Multi-pipeline object-detection system + Docker inference API
> **Status:** Functional but iteration-heavy / experimental; Docker image deployable
> **Folder:** `task3/`

---

## 1. Overview

Task 3 is a production object-detection system for **pharmaceutical carton / packaging artwork recognition**. It detects an **8-class structural ontology** on carton images and feeds downstream OCR/Rx pipelines. The project spans data engineering (COCO→YOLO conversion, augmentation, negative-sample injection), training (**YOLOv8** + **RF-DETR**), **hard-example mining**, geometric **structural completion** to recover missed panels, and a **Dockerized Flask inference API** repeatedly repackaged for a GPU server.

## 2. Problem Statement

Managing thousands of packaging images requires clean, leak-free datasets and strict ontology enforcement (no orphaned labels, stable class IDs), while detectors must localize artwork and every panel (Front/Side/Back/Bottom/Top) reliably. When detectors miss a panel, a geometric reasoner recovers it instead of failing silently.

## 3. Architecture & Modules

**Root inference / ops scripts**
- `app.py` — **Flask** API (`/health`, `/predict`, `/batch`, `/info`) wrapping `ultralytics.YOLO`; class map 0–7; env-driven `MODEL_PATH` (default `/app/models/checkpoint_best_total.pth`), `CONF_THRESHOLD=0.25`, `IOU_THRESHOLD=0.7`, `IMG_SIZE=512`.
- `rfdetr_detector.py` — `Detection` class wrapping `rfdetr.RFDETRNano` (2-class, imgsz 640, 50 queries); `detect()` returns boxes/conf/class.
- `run_inference.py` — batch YOLO → YOLO txt + `inference_results.json`.
- `run_blind_inference.py` — RF-DETR blind test on `final_training/images`.
- `run_hard_example_mining.py` — drives `hard_example_pipeline`.
- `prepare_final_dataset.py` — pad/shift + 5 augmentations (orig/gray/rot180/rot90cw/ccw), 70/20/10 split by original ID (seed 42), writes COCO.
- `split_dataset.py`, `build_package.py`, `package_for_server.py`, `zip_images.py`, `visualize_10_images.py`, `visualize_all_images.py`, `combine_eval.py` (COCO merge + category remap), `test_rfdetr_load.py` / `test_rfdetr_predict.py` / `test_reasoner.py` (smoke tests).

**`src/`**
- `data/` — `coco_to_yolo.py` (negative / scraped injection), `create_yaml.py`, `augment.py`.
- `pipeline/` — `prepare_yolo_dataset.py`, `train_8class_{20,30,50,feasibility,baseline,round_2,sanity}.py`, `rx_localization_pipeline.py`, `run_rx_detection.py`, `validate_v2.py`.
- `tools/` (~35) — `dataset_audit.py`, `convert_to_8class.py`, `clean_front_back.py` (filename-hash dedup), `verify_negative_samples.py`, etc.

**`structural_completion/`** — the most original component:
- `completion_engine.py` — `CompletionEngine.process()` builds a `PanelGraph` from detections (`Panel` nodes, `add_edge`), crops artwork (`ArtworkCropper`), extracts geometry (`RectangleDetector`: Canny+Hough+connected components; `RectangleFilter`: area/aspect/redundancy rejection), then runs an **iterative loop** (`max_iterations=3`) where `Validator.get_missing_classes()` finds gaps, `PanelAssignment.assign()` scores `Matcher` candidates (size similarity, shared-edge length, center distance) against layout-rule anchors, adding inferred `source=geometry_reasoner` panels, ending with `SanityChecker` + `FailureReporter` (`unrecovered_panels.jsonl`).
- `reasoning/layout_rules.py` loads `configs/layout_rules.json` (required/optional/adjacency); `visualization/visualizer.py` draws dashed blue boxes for inferred panels.

**`hard_example_pipeline/`** — `classifier.py` (`FailureClassifier`: IoU-based missing/duplicate/false_positive/low_confidence), `collector.py` (`HardExampleCollector` JSONL DB), `statistics.py`, `dataset_report.py` (`DatasetReportGenerator` → `hard_examples/dataset_report.md`), `visualizer.py`, `export.py`.

**Training pipelines** — `rfdetr_training_pipeline/scripts/` (`pipeline.py` `RFDETRPipeline.predict_carton`, `train.py` w/ experiment.yaml + git/env capture, `hard_example_mining.py`, `benchmark.py`, `evaluate*.py`, `yolo_to_coco.py`); `final_training_pipeline/` (2-class Artwork/not_artwork); `final_training_v2/`; `final_training_package/` (weights/, `gpu_waiter`/`start_training`); `server_training_package_v2/`.

## 4. Core Techniques

- **Dual detectors** — YOLOv8 (Ultralytics, Nano/m, imgsz 512) + RF-DETR (Nano/Small, imgsz 640).
- **Dataset engineering** — COCO↔YOLO conversion, bbox-aware augmentation & rotation, dynamic negative-sample injection from `Scrapped/output/images`.
- **8-class ontology** — `Artwork, not_artwork, Front_Panel, Side_Panel, Back_Panel, Extra_Areas, Bottom_Panel, Top_Panel`.
- **Structural completion** — geometric reasoning recovers missed panels via layout rules + candidate matching.
- **Hard-example mining** — failure taxonomy → re-annotation batches.
- **Containerization** — Docker + docker-compose for the Flask inference API.

## 5. Data & Artifacts

- Datasets: `images/`, `annotations/instances_Train.json`, `config/yolo_dataset.yaml`, Roboflow `train/valid/test` COCO (`_annotations_mapped.json`), `final_training_v2/{train,valid,test}` (many `.rf.*` filenames).
- Checkpoints: `output/models/checkpoint_best_total*.pth`, `output/*.ckpt` / `*.pth`, `final_training_package/weights/`.
- Packages: `server_training_package.zip` (~6.6 GB), `server_training_package_v2.zip` (~7.3 GB), `visualized_images.zip`.
- Reports: `hard_examples/dataset_report.md` (5 hard examples, avg conf 0.690, top failures `missing_back`/`white_pharma`), `class_distribution_histogram.png`, `failure_reports/unrecovered_panels.jsonl`.

## 6. Tech Stack / Dependencies

`requirements.txt`: ultralytics, torch, torchvision, opencv, albumentations, pandas, flask, scikit-learn, … `Dockerfile` (python:3.10-slim, copies `app.py`+`run_inference.py`, HEALTHCHECK, EXPOSE 5000). `docker-compose.yml` (service `ml-inference`, mounts `./trained_models`, `MODEL_PATH=.../checkpoint_best_total.pth`). `pyrightconfig.json`. Two venvs (`.venv`, `.venv_rfdetr`) and `wheels/` suggest offline installs.

## 7. Maturity / Status

**Functional but iteration-heavy / experimental.** Multiple parallel pipelines (`src` YOLO, `rfdetr_training_pipeline`, `final_training`, `final_training_v2`, `final_training_package`) and many `train_*` variants indicate active tuning, not a single frozen line. Some **stubs**: `rfdetr_training_pipeline/scripts/benchmark.py` and `hard_example_mining.py` print placeholder results; `run_hard_example_mining.py` uses **mock ground truth** and only the first 5 images. Class naming inconsistent (`Bottom_Part` vs `Bottom_Panel`). **Windows-path-bound** (hardcoded `d:\Silvertouch Internship\task3`) — not directly portable to Linux. Docker YOLO API is deployable; RF-DETR path needs `rfdetr` + `.venv_rfdetr`.

> 📝 **Correction vs. parent README:** the README describes a perceptual-hashing (pHash) scraper for de-duplication. In-repo, dedup is **filename-hash based** (`clean_front_back.py`), and "scraping" refers to a pre-existing `Scrapped/output/images` negatives folder — there is **no scraper script** in this repo. The README's `secure_dataset_merge` orphan-gate snippet is conceptual, not a 1:1 match to the actual `src/data/coco_to_yolo.py` logic.

## 8. How to Run

```bash
# Docker (YOLO inference API):
docker build -t ml-model-inference:latest .
docker compose up --pull always
curl http://localhost:5000/health
curl -X POST -F "image=@img.jpg" http://localhost:5000/predict

# YOLO training:
python scripts/train_final.py                      # final_training_pipeline
python final_training_package/scripts/start_training.py   # GPU server (or gpu_waiter.py)

# RF-DETR:
python rfdetr_training_pipeline/scripts/train.py   # / predict.py

# Dataset prep:
python src/pipeline/prepare_yolo_dataset.py        # uses config/yolo_dataset.yaml

# Hard mining (currently mock / 5-image demo):
python run_hard_example_mining.py

# Structural completion (library use):
#   CompletionEngine.process(image, detections)  from structural_completion
```

**Key files:** `app.py`, `rfdetr_detector.py`, `structural_completion/completion_engine.py`, `hard_example_pipeline/classifier.py`, `src/data/coco_to_yolo.py`, `config/yolo_dataset.yaml`, `Dockerfile`, `docker-compose.yml`.

---

*Report generated from source inspection of `task3/`. See parent `README.md` for the portfolio-level summary of this task. See also in-repo `README.md`, `CONTAINER_SETUP.md`, `final_training_pipeline/README.md`, `final_training_package/README.md`, `hard_examples/dataset_report.md`.*
