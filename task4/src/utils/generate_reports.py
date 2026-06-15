import os
import sys

# Add project root to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from src.utils.file_ops import save_markdown_artifact

arch_content = """# Architecture Report: Enterprise Semantic Retrieval Layer

## Overview
The retrieval layer is designed to orchestrate the semantic resolution of natural language queries into the components required for accurate Text-to-SQL generation.

## Components
1. **Entity Extractor (`src/routing/extractor.py`)**: Uses the `semantic_catalog.json` to extract concepts, locations, and institution types from raw text.
2. **Module Router (`src/routing/router.py`)**: Predicts the target database schema module based on extracted entities and intent keywords, supporting multi-label classification.
3. **Retrievers (`src/retrieval/`)**:
   - `doc_retriever`: Fetches business rules mapped to the predicted module.
   - `ddl_retriever`: Fetches physical schemas, injecting geographical masters if locations are detected.
   - `sql_retriever`: Uses a baseline `TFIDFRetriever` (extending `BaseRetriever`) to locate semantically similar historical questions.

## Future Proofing
The `BaseRetriever` abstraction ensures that migrating from local TF-IDF to a Vector Database (e.g. Qdrant) or a neural embedding model (e.g. text-embedding-ada-002) requires zero refactoring of the Context Assembler orchestration layer.
"""

eval_content = """# Evaluation Report: Phase 1 Baseline

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
"""

def generate_reports():
    reports_dir = os.path.join(project_root, 'outputs', 'reports')
    save_markdown_artifact(arch_content, reports_dir, 'architecture_report')
    save_markdown_artifact(eval_content, reports_dir, 'evaluation_report')
    print("Reports generated.")

if __name__ == "__main__":
    generate_reports()
