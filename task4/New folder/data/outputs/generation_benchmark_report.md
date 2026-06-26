# SQL Generation Benchmark Report

## Retrieval Metrics
* **Candidate Recall**: 88.00%
* **DDL Recall**: 70.00%
* **Context Sufficiency**: 70.00%

## Generation Metrics
* **Exact Match**: 0.00%
* **Structural Match**: 0.00%
* **Validation Pass Rate**: 0.00%

## Hallucination Metrics
* **Table Hallucination Rate**: 0.00%
* **Column Hallucination Rate**: 0.00%
* **Join Hallucination Rate**: 0.00%

## Complexity Breakdown (Structural Accuracy)
* **AGGREGATION**: 0.00% (0/21)
* **FILTER_QUERY**: 0.00% (0/11)
* **SUBQUERY**: 0.00% (0/6)
* **RANKING**: 0.00% (0/2)
* **MULTI_JOIN**: 0.00% (0/10)

## Failure Breakdown
* **VALIDATION_FAILURE**: 44
* **RETRIEVAL_FAILURE**: 6

## Recommendation
**Continue improving Retrieval, Prompting, or Validation.**

*Rationale*: Retrieval thresholds were not met, or generation accuracy is already sufficient. The model is not the primary bottleneck.
