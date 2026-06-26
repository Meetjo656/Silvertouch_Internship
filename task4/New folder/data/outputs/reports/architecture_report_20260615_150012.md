# Architecture Report: Enterprise Semantic Retrieval Layer

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
