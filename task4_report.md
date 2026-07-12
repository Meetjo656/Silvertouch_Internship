# Task 4 — Enterprise Semantic SQL RAG & LLM Tuning

> **Domain:** RAG / LLM Fine-Tuning / Text-to-SQL / Information Retrieval
> **Type:** Research pipeline + experiment harness + web studio
> **Status:** Research / prototype stage (active iteration)
> **Folder:** `task4/`

---

## 1. Overview

Task 4 builds a **RAG pipeline that turns natural-language questions into enterprise SQL** over an Indian-government / banking domain (modules such as `GP_COVERAGE`, `COOPERATIVE`, `BANKS`, `EDUCATION`). It combines (a) **semantic schema + business-concept retrieval** over FAISS, (b) **context assembly / optimization** for prompt construction, and (c) **QLoRA fine-tuning of Qwen2.5-Coder** to improve SQL generation. The large `scripts/` surface and `outputs/` artifacts reflect active experimentation (curriculum building, ablation studies, retrieval-gap audits) rather than a shipped product.

## 2. Problem Statement

LLMs hallucinate SQL when they don't know the exact database schema. Standard vector search fails on large schemas. The solution: a hierarchical **semantic router** that embeds DDL + business concepts with SentenceTransformers and indexes them in **FAISS**, then assembles only the relevant schema/SQL context into the prompt — and fine-tunes Qwen with LoRA so generated SQL is accurate and faithful.

## 3. Architecture & Modules (`src/`)

| Subpackage | Notable files | Responsibility |
|---|---|---|
| `context` | `user_context.py` (`UserContext`) | Per-user filters / role |
| `data_pipeline` | `build_dataset.py`, `load_schema.py`, `show_head.py` | Schema loading & dataset prep |
| `evaluation` (15) | `benchmark_runner.py` (`BenchmarkRunner`), `evaluator.py`, `execution_match.py`, `structural_match.py` (`StructuralMatch`), `sql_comparator.py` (`SQLComparator`), `failure_analyzer.py`, `hallucination_detector.py`, `query_complexity_classifier.py`, `evaluate_routers.py`, `benchmark_report.py` | SQL correctness / failure analysis |
| `generation` | `qwen_generator.py` (`QwenGenerator`, **OpenAI-compatible client**), `prompt_builder.py`, `prompt_builder_strict.py`, `context_assembler.py`, `context_optimizer.py` (scores/prunes tables), `column_selector.py`, `generation_runner.py`, `generation_benchmark.py` | Prompt → generate → validate |
| `ingestion` | `build_corpus.py`, `profiler.py` | Build retrieval corpus & profile tables |
| `retrieval` | `embedding_retriever.py` (`EmbeddingRetriever`, MiniLM + FAISS `fit`/`search`), `ddl_retriever.py`, `doc_retriever.py`, `sql_retriever.py`, `sql_guided_semantic_retriever.py` (join-frequency expansion), `context_ranker.py` (`ContextRanker`, weighted 0.50/0.20/0.15/0.10/0.05 re-ranking), `schema_matcher.py` | Dense retrieval + re-ranking |
| `routing` | `router.py` (`ModuleRouter`), `retrieval_router.py`, `semantic_router.py`, `extractor.py`, `build_module_embeddings.py` (→ `data/embeddings/module_embeddings.npy`) | Route query to module |
| `schema` | `schema_registry.py` (`SchemaRegistry`: authoritative + inferred DDLs, coverage, missing-table detection) | Authoritative schema store |
| `semantic` | `business_semantics_resolver.py`, `join_path_resolver.py` (join-column discovery + fallback), `enum_extractor.py` | Inject business meaning + join paths |
| `training` | `train_qlora_usloth.py` (`main()`), `check_dataset_tokens.py` | QLoRA fine-tuning |
| `utils` | `file_ops.py`, `generate_reports.py`, `sql_structure_extractor.py` | Helpers |
| `validation` | `sql_validator.py` (`SQLValidator`, sqlglot-based validity vs registry) | SQL validity |

**`scripts/` categories (30+)**
- *Catalog/builders*: `build_business_semantics_catalog.py`, `build_schema_signature_catalog.py`, `build_query_pattern_library.py`, `reconstruct_missing_schema.py`.
- *Dataset / LoRA / curriculum*: `build_ft_dataset.py`, `build_lora_dataset.py`, `build_join_curriculum.py`, `build_training_curriculum.py`, `analyze_dataset.py`, `analyze_training_dataset.py`, `mine_filter_patterns.py`, `process_sql_knowledge.py`, `patch_dataset.py`.
- *Audits*: `audit_dataset.py`, `audit_join_coverage.py`, `audit_lora_dataset.py`, `schema_coverage_audit.py`, `retrieval_gap_analysis.py`, `context_loss_analysis.py`, `debug_prompt_samples.py`, `debug_validator.py`.
- *Retrieval evals*: `retrieval_recall_curve.py`, `evaluate_sql_guided_retrieval.py`.
- *Experiments / entry*: `experiment_runner.py` (`run_experiment`), `experiment_combined.py`, `experiment_context_reduction.py`, `experiment_strict_prompt.py`, `evaluate_generation.py`.

## 4. Key Entry Points

- **Training:** `python src/training/train_qlora_usloth.py`.
- **Benchmark / eval:** `python scripts/experiment_runner.py`, `scripts/evaluate_generation.py` (uses `GenerationBenchmarkPipeline`).
- **`studio/`** — a **web UI / experiment viewer** backed by `studio.db` (SQLite) + `auth/auth.db` and a bootstrap password; contains `assets`, `cache`, `exports`, `outputs`, `runs`, `studio.pid`. Served app (FastAPI per deps) for browsing experiment runs; no app `.py` is in-tree (likely served separately). `runs/` is currently empty.

## 5. Core Techniques

- **FAISS dense retrieval** + SentenceTransformers (`all-MiniLM-L6-v2`) via `EmbeddingRetriever.fit/search`; module-router embeddings in `data/embeddings/`.
- **Semantic schema catalog** — `enterprise_semantic_catalog.json` (concepts→modules→tables/columns, parent/child, aliases, confidence). `BusinessSemanticsResolver` + `JoinPathResolver` inject business meaning and join paths into prompts.
- **QLoRA fine-tuning** — `train_qlora_usloth.py` loads `Qwen/Qwen2.5-Coder-7B-Instruct` via `unsloth.FastLanguageModel` (4-bit `load_in_4bit`, `get_peft_model`, `r=32, alpha=64, dropout=0.05`, all attention+MLP proj targets), `SFTTrainer`, `max_seq_length=2048`, 3 epochs, lr `2e-4`, batch 2 × grad-accum 8; saves **LoRA adapters only**.
- **Training curriculum** — `build_training_curriculum.py` mixes `raw_qwen_lora_dataset.jsonl` + `synthetic_join_dataset.jsonl`; `build_join_curriculum.py` / `build_lora_dataset.py` build reasoning subsets (`filter_reasoning_subset`, `join_reasoning_subset`).
- **Context optimization** — `ContextOptimizer` + `ContextRanker` prune/rank DDLs, docs, SQL examples before prompting.

## 6. Tech Stack / Dependencies

`requirements.txt`: transformers, datasets, accelerate, evaluate, peft, trl, bitsandbytes, sentencepiece, tiktoken, pandas/numpy/pyarrow, pyyaml/omegaconf, tensorboard, wandb, scikit-learn, sqlparse, sqlalchemy, **fastapi + uvicorn**, **sqlglot**. Tests: `tests/test_*.py` (4 files, **all empty stubs**); `test_unsloth.py` at root is a smoke check.

## 7. Data & Artifacts

`data/` → `raw/` (Private/Public Data, banks_sample_csv), `processed/` (`qwen_ft_dataset.json`, `schema_signature_catalog.json`, `join_graph.json`, `query_pattern_library.json`, `filter_pattern_library.json`, `train.jsonl`), `training/` (`qwen_lora_train_v2.jsonl`, `qwen_lora_validation_v2.jsonl`, `synthetic_join_dataset.jsonl`, reasoning subsets), `benchmark/` (`retrieval_benchmark_*.json`), `embeddings/` (`module_embeddings.npy`, `module_metadata.json`), `semantic_catalog/`, `outputs/`. Plus `enterprise_semantic_catalog.json`, `.cache/huggingface`, `unsloth_compiled_cache/`.

## 8. Maturity / Status

**Research / prototype stage.** Rich, well-factored modules and a large experimentation harness, but **no docs/ (empty)**, **tests are empty stubs**, `studio/runs` empty, and `qwen_generator.py` depends on an **external OpenAI-compatible endpoint** (not bundled). Clearly actively iterated (many audit/analysis scripts). Not a turnkey product.

## 9. How to Run (inferred)

```bash
pip install -r requirements.txt
# Serve Qwen via an OpenAI-compatible endpoint (e.g. vLLM) for QwenGenerator
python src/training/train_qlora_usloth.py     # fine-tune LoRA adapters
python scripts/build_lora_dataset.py          # build dataset
python scripts/experiment_runner.py           # run benchmark
python scripts/experiment_combined.py         # combined-prompt experiment
python scripts/evaluate_generation.py         # generation eval
```

> 📝 **Correction vs. parent README:** the README shows a `SchemaDenseRetriever` using `IndexFlatL2` and an Unsloth `Qwen2.5-7B-Instruct` setup with `r=16`. The actual code uses **`Qwen2.5-Coder-7B-Instruct`** with **`r=32, alpha=64`** and a `ContextRanker` weighted re-ranker — functionally aligned but parameter/name details differ. The README code is illustrative.

---

*Report generated from source inspection of `task4/`. See parent `README.md` for the portfolio-level summary of this task.*
