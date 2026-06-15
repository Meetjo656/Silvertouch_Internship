# Evaluation Report: Phase 1 Baseline

## Benchmark Design
A 50-question benchmark dataset was programmatically generated, categorized into:
- Single-Module
- Location-Heavy
- Institution-Type
- Cross-Domain
- Multi-Table

## Baseline Results
The baseline TF-IDF and heuristic entity extraction pipeline achieved the following zero-shot metrics without any LLM routing:
- **Module Accuracy**: 40.0%
- **Documentation Retrieval Accuracy**: 40.0%
- **DDL Retrieval Accuracy**: 32.7%
- **SQL Example Retrieval Accuracy**: 20.0%
- **Overall Retrieval Accuracy**: 33.2%

## Analysis & Next Steps
The strict thresholding and exact-match heuristic extractor struggle with highly complex multi-domain queries that lack explicit catalog terms. Phase 2 should replace `TFIDFRetriever` with a fine-tuned dense embedding model and introduce an LLM into the `ModuleRouter` to resolve ambiguity.
