# Task 5 — MyBot Custom NLU Microservice

> **Domain:** MLOps / Microservices / NLP
> **Type:** FastAPI NLU service (intent + entity extraction) + offline experiments
> **Status:** Mature & well-documented; runnable (has `.venv`)
> **Folder:** `task5/`

---

## 1. Overview

Task 5 is a standalone **FastAPI microservice** providing two-model NLU for the "MyBot" workflow engine: **intent / sub-intent detection** and **entity extraction**. It sits between a chat client and a Node/Next.js workflow engine (`EntryNodeHandler`), routing user messages to the correct workflow subflow. It supports per-workflow supervised training, incremental retraining, conversation/session management, feedback-driven active learning, and a benchmark suite. `main.py` is at **v3.2 — "Per-Workflow Multi-Model with spaCy"**.

## 2. Problem Statement

Heavy monolithic models are too slow for instant chat intents. The service needs a blazingly fast, lightweight NLU layer that can be trained per-workflow, cached to avoid retraining, and embedded in a larger conversational workflow engine.

## 3. Architecture & Modules (`app/`)

| Subpackage | Notable files | Responsibility |
|---|---|---|
| root | `main.py` (`FastAPI` app v3.2.0), `config.py` (env dirs), `state.py` (runtime singletons: `spacy_nlp`, `intent_model`, `trained_classifiers`, `trained_entity_classifiers`), `models.py` (Pydantic schemas), `auth.py` (`get_api_key`) | App wiring |
| `core/` (913 LOC) | `cache.py` (`.npz` load/save/clear), `fingerprint.py` (`intent_fingerprint` — **MD5**), `model_registry.py` (`ModelRegistry` → `model_registry.json`), `persistence.py` (save/load `.joblib`, backups/rollbacks), `knowledge_graph.py` (O(1) lookup), `evaluation.py` (80/20, `LogisticRegression`), `augmentation.py` | Core services |
| `training/` | `intent.py` (`train_intent_classifier`, `train_intent_classifier_incremental`, `train_global_classifiers`, `_fit_classifier_and_evaluate`), `entities.py` (per-entity binary `LogisticRegression`) | Supervised training |
| `inference/` | `intent.py` (`detect_intent`), `entities.py` (`extract_entities`), `validation.py`, `fallback.py` (`manage_fallback`) | Runtime inference |
| `conversation/` (452 LOC) | `manager.py` (`ConversationManager`), `session.py`, `resolver.py` (`ResolutionEngine`: `should_classify`/`ask_entity`/`fallback`), `validators.py` (`EntityValidator`) | Multi-turn dialog |
| `hierarchy/` (323 LOC) | `workflow_classifier.py`, `workflow_resolver.py`, `hierarchy_manager.py`, `candidate_filter.py`, `intent_ranker.py` | Workflow routing |
| `ranking/` (268 LOC) | `intent_ranker.py`, `score_combiner.py` (`ScoreCombiner.combine`), `confidence_calibarator.py` (accept ≥0.75 / clarify ≥0.50) | Confidence handling |
| `feedback/` (319 LOC) | `collector.py` (`FeedbackCollector`), `active_learning.py` (uncertainty queue), `dataset_builder.py` | Active learning |
| `benchmark/` (643 LOC) | `benchmark.py` (`BenchmarkRunner`), `accuracy.py`, `latency.py`, `memory.py`, `scalability.py`, `stress.py`, `comparison.py`, `profiler.py`, `report_generator.py`, `visualiser.py` | Perf/benchmark suite |
| `routers/` (502 LOC) | `train.py` (`POST /train`, `/train/incremental`), `analyze.py` (`POST /analyze`, `/detect-intent`, `/extract-entities`), `status.py` (`/health`, `/training-status`) | HTTP endpoints |

Root scripts: `train.py` (CLI: `--workflow/--all/--list/--status/--reset/--augment/--test`, POSTs to a running service), `insert_nlu_routers.py` (injects NLU router nodes into child workflow JSON). `experiments/` holds an offline validation harness (`analysis.py`, `evaluation.py`, `experiment_runner.py`, `dataset_builder.py`, `error_analysis.py`, `architecture_validation.py`, `report.py` + `.md/.json/.txt` reports).

## 4. Key Entry Points

- **`main.py`** — builds the FastAPI app; `lifespan()` loads spaCy `en_core_web_sm` + `sentence-transformers/all-MiniLM-L6-v2`; auto-restores trained classifiers via `ModelRegistry`; registers 3 routers; CORS open. **Defaults to port 8001.**
- **`train.py`** — CLI that trains workflows by POSTing to a running service at `http://127.0.0.1:8000`.
- **`insert_nlu_routers.py`** — utility injecting NLU router nodes into child workflow JSON (refs `mybot_frontend/public/workflows`).

## 5. Core Techniques — IMPORTANT CORRECTION

> 📝 **The parent README's stated stack does NOT match the actual code.** The real pipeline:
> - **Intents:** **SentenceTransformer embeddings (`all-MiniLM-L6-v2`) + `sklearn.LogisticRegression`** (NOT TF-IDF + LinearSVC). Zeroshot fallback uses cosine similarity.
> - **Entities:** **spaCy `en_core_web_sm` NER + regex patterns (AppID/Phone/Email/Date) + per-entity binary `LogisticRegression`** (NOT CRF).
> - **Caching:** **MD5** fingerprints in `fingerprint.py` (NOT SHA-256); embeddings cached as `.npz` (`cache.py`).
> - **Model registry:** `core/model_registry.py` → `trained_models/model_registry.json`; per-workflow `.joblib` via `joblib.dump`.
> - **Ranking / hierarchy:** `ScoreCombiner`, `IntentRanker`, `ConfidenceCalibrator` combine workflow/intent/entity/context scores with dynamic thresholds.
>
> The offline `experiments/` harness **does** use `TfidfVectorizer + LogisticRegression` (matching the README's scikit-learn hint) — but it's a validation harness, not the service.

## 6. Tech Stack / Dependencies

`requirements.txt`: fastapi, uvicorn, transformers, torch, sentence-transformers, numpy<2.0, pydantic, python-dotenv, scikit-learn, requests, spacy + `en_core_web_sm`, matplotlib. `pyrightconfig.json` present (type-checking). `Frontend/` is a **static demo** (`templates/chatbot.html` + `styles/main.css`) — simple HTML/CSS chatbot, NOT a Next.js app despite README's integration notes.

## 7. Data & Artifacts

- `training_data/*.json` — 13 workflow files (`welcome-intent-anonymous`, `portal-user-guides-*`, `application-status-anonymous`, `aatest*`, plus versioned `_v1..v9`).
- `trained_models/*.joblib` — 11 trained classifiers + `model_registry.json`.
- `embedding_cache/*.npz` — `aatest`, `application-status-anonymous`, `welcome-intent-anonymous`.
- `experiments/datasets/{original,merged,benchmark}`, `experiments/baseline_results.json`, `new_results.json`, plus `.md/.txt` reports.

## 8. Maturity / Status

**Mature and well-documented.** Rich docs: `report.md` (137 KB auto-generated AST deep-dive), `report.pdf`, `core_modules_explanation.md` (higher-quality narrative), `INSTALL_SPARCY.md`, `research_report.md`. Note: `architecture_deep_dive.md` is **empty (0 bytes)** — placeholder not filled. `report.md` is a generic generated dump; `core_modules_explanation.md` is the better narrative read. Runnable (has `.venv`).

> ⚠️ **Port mismatch:** the service defaults to **8001** in `main.py`, but `train.py` (CLI) targets **8000** (`NLU_URL`). Align these before training via the CLI.

## 9. How to Run

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
# or: python main.py  (Windows: start.bat)
curl http://localhost:8001/health
python train.py --all        # train all workflows
python train.py --status     # view CV accuracy
```

---

*Report generated from source inspection of `task5/`. See parent `README.md` for the portfolio-level summary of this task. For the authoritative narrative, prefer `core_modules_explanation.md` over the auto-generated `report.md`.*
