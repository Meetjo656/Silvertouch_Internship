# Silvertouch Internship — Project Documentation

Technical documentation for the five projects in this repository. Detailed per-project reports are available at `taskN_report.md`.

---

## 1. Task 1 — Perceptual Text-Color Comparison

**Domain:** Computer Vision / Color Science
**Status:** Complete, runnable

**Purpose.** Determine whether the text color of two cropped packaging images (reference vs. target) is an acceptable perceptual match.

**Approach.**
- Foreground isolation via CIE 1976 ΔE against a border-pixel background estimate.
- Dominant color extraction by MeanShift clustering in CIE L\*a\*b\* space (`sklearn.cluster.MeanShift`, dynamic bandwidth).
- Color naming via nearest-neighbor search over the 949-color XKCD palette (CIEDE2000).
- Match decision via CIEDE2000 ΔE; threshold 8.0 (`colour.delta_E`).

**Key entry point.** `main.py` → `main()`; reads `data/reference.png`, `data/target.png`; writes `outputs/comparison_dashboard.png`.

**Stack.** OpenCV, NumPy, scikit-learn, scikit-image, colour-science, Matplotlib.
**Note.** No `requirements.txt`; dependencies undocumented.

**Run.**
```
pip install opencv-python numpy scikit-learn scikit-image colour matplotlib
python main.py
```

---

## 2. Task 2 — Structural Packaging Intelligence

**Domain:** Geometric Machine Learning
**Status:** Research WIP, partially runnable

**Purpose.** Reconstruct the structural layout (panels, flaps, tabs) of packaging blueprints from images, with topologically valid, non-overlapping geometry.

**Approach.**
- Hough line detection (`cv2.HoughLinesP`) → Planar Straight Line Graph via Shapely `unary_union` (`PSLGBuilder`).
- Polygonization into closed faces (`Polygonizer`); refinement across a 7-phase pipeline.
- 17 geometric features per face (`extract_geometric_features`); classification by Random Forest (`StructuralMaskRFClassifier`, scikit-learn + StandardScaler).
- Taxonomy: PANEL, FLAP, GLUE_TAB, DUST_FLAP, BACKGROUND, UNKNOWN.

**Key entry points.** `src/pipeline/geometry_pipeline.py` (Stages 0–4); `src/training/train.py`; `generate_and_ingest.py` (external SAM dependency); `analyze_masks.py`.

**Stack.** OpenCV, NumPy, pandas, scikit-learn, scikit-image, Shapely, Pydantic, Matplotlib, Seaborn, Joblib.
**Notes.** `shapely` absent from `requirements.txt`; depends on an out-of-repo `task2` SAM module; `tests/` empty.

**Run.**
```
pip install numpy pandas opencv-python scikit-learn pydantic matplotlib seaborn joblib shapely
python -m src.pipeline.geometry_pipeline
python -m src.training.train --n_estimators 500
```

---

## 3. Task 3 — Packaging & Panel Detection Pipeline

**Domain:** Computer Vision / MLOps
**Status:** Functional, iteration-heavy; Docker API deployable

**Purpose.** Detect an 8-class structural ontology (Artwork, not_artwork, Front/Side/Back/Bottom/Top_Panel, Extra_Areas) on carton images for downstream OCR/Rx pipelines.

**Approach.**
- Dual detectors: YOLOv8 (Ultralytics, imgsz 512) and RF-DETR (Nano, imgsz 640).
- Dataset engineering: COCO↔YOLO conversion, augmentation, negative-sample injection; 70/20/10 split (seed 42).
- Structural completion: geometric reasoner recovers missed panels via layout rules + candidate matching (`CompletionEngine`, iterative, max 3 passes).
- Hard-example mining: IoU-based failure taxonomy → re-annotation (`hard_example_pipeline`).
- Dockerized Flask inference API (`app.py`).

**Key entry points.** `app.py` (Flask `/predict`); `rfdetr_detector.py`; `run_inference.py`; `prepare_final_dataset.py`; `structural_completion/completion_engine.py`; `run_hard_example_mining.py`.

**Stack.** Ultralytics, PyTorch, OpenCV, Albumentations, pandas, Flask, scikit-learn. Docker, docker-compose.
**Notes.** Some mining/benchmark scripts are stubs; Windows-path-bound; class naming inconsistent (`Bottom_Part` vs `Bottom_Panel`). De-duplication is filename-hash based (no pHash scraper in-repo).

**Run.**
```
docker build -t ml-model-inference:latest .
docker compose up --pull always
curl -X POST -F "image=@img.jpg" http://localhost:5000/predict
# Training:
python src/pipeline/prepare_yolo_dataset.py
python rfdetr_training_pipeline/scripts/train.py
```

---

## 4. Task 4 — Enterprise Semantic SQL RAG & LLM Tuning

**Domain:** RAG / Text-to-SQL / LLM Fine-Tuning
**Status:** Research prototype, active iteration

**Purpose.** Convert natural-language questions into enterprise SQL over government/banking schemas, minimizing hallucination.

**Approach.**
- Dense retrieval: SentenceTransformers (`all-MiniLM-L6-v2`) + FAISS (`EmbeddingRetriever`); hierarchical semantic router (`ContextRanker`, weighted re-ranking).
- Semantic catalog: `enterprise_semantic_catalog.json` + `BusinessSemanticsResolver` + `JoinPathResolver` inject schema/ business meaning into prompts.
- Context optimization: `ContextOptimizer` prunes/ranks DDL, docs, SQL examples.
- QLoRA fine-tuning: Qwen2.5-Coder-7B via Unsloth (`r=32, alpha=64`, 4-bit, SFTTrainer, 3 epochs); LoRA adapters only.
- Web studio (`studio/`, SQLite-backed) for experiment browsing.

**Key entry points.** `src/training/train_qlora_usloth.py`; `scripts/experiment_runner.py`; `scripts/evaluate_generation.py`.

**Stack.** Transformers, PEFT, TRL, Unsloth, bitsandbytes, SentenceTransformers, FAISS, sqlglot, FastAPI, pandas, WandB/TensorBoard.
**Notes.** `tests/` are empty stubs; `qwen_generator.py` requires an external OpenAI-compatible endpoint.

**Run.**
```
pip install -r requirements.txt
python src/training/train_qlora_usloth.py
python scripts/experiment_runner.py
```

---

## 5. Task 5 — MyBot NLU Microservice

**Domain:** MLOps / NLP Microservices
**Status:** Mature, well-documented, runnable

**Purpose.** FastAPI service for per-workflow intent/sub-intent detection and entity extraction, routing chat messages to a workflow engine.

**Approach.**
- Intents: SentenceTransformer (`all-MiniLM-L6-v2`) embeddings + `LogisticRegression`; zeroshot cosine fallback.
- Entities: spaCy `en_core_web_sm` NER + regex patterns (AppID/Phone/Email/Date) + per-entity binary `LogisticRegression`.
- Caching: MD5 fingerprinting (`fingerprint.py`); embeddings cached as `.npz`; model registry (`model_registry.json`).
- Conversation: multi-turn session/resolver, feedback-driven active learning, benchmark suite.
- Confidence handling: `ScoreCombiner` + `ConfidenceCalibrator` (accept ≥0.75, clarify ≥0.50).

**Key entry points.** `main.py` (FastAPI, port **8001**); `train.py` (CLI, targets port **8000**); `insert_nlu_routers.py`.

**Stack.** FastAPI, Uvicorn, SentenceTransformers, spaCy, scikit-learn, Torch, Pydantic.
**Notes.** `architecture_deep_dive.md` is empty. Port mismatch between service (8001) and `train.py` CLI (8000).

**Run.**
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
python train.py --all
```

---

*Generated from `task1_report.md`–`task5_report.md`. For full detail, consult the per-project reports.*
