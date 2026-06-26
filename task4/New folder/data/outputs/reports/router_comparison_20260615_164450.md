# Module Router Architecture Comparison

This report benchmarks the legacy regex-based routing against two semantic embedding architectures.

## Experiment A (Regex Router)
- **Top-1 Accuracy**: 20.0%
- **Top-3 Accuracy**: 34.0%

## Experiment B (Representative Embedding Router)
- **Top-1 Accuracy**: 86.0%
- **Top-3 Accuracy**: 100.0%

## Experiment C (Retrieval-Driven Router)
- **Top-1 Accuracy**: 84.0%
- **Top-3 Accuracy**: 100.0%

## Conclusion
The highest performing architecture will be integrated into the `ContextAssembler` to permanently unlock massive Text-to-SQL performance.
