# Phase 7.5 Report: Live Qwen Generation Benchmark

## Overall Metrics
* **Exact Match**: 38.00%
* **Structural Match**: 44.00%
* **Validation Pass Rate**: 82.00%

## Hallucination Metrics
* **Table Hallucination Rate**: 12.00%
* **Column Hallucination Rate**: 18.00%

## Failure Breakdown
* **FILTER_FAILURE**: 44.00% (22/50)
* **RETRIEVAL_FAILURE**: 4.00% (2/50)
* **SYNTAX_FAILURE**: 4.00% (2/50)
* **TABLE_SELECTION_FAILURE**: 4.00% (2/50)

## Complexity Breakdown
* **AGGREGATION**: 52.38% (11/21)
* **FILTER_QUERY**: 54.55% (6/11)
* **MULTI_JOIN**: 0.00% (0/10)
* **SUBQUERY**: 83.33% (5/6)
* **RANKING**: 0.00% (0/2)

