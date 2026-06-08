# Evaluation Protocol

To ensure the structural detection model performs reliably in production, evaluation must be rigorous.

## Required Metrics
Model evaluation MUST include the extraction of:
- **Precision**
- **Recall**
- **mAP50** (Mean Average Precision at IoU 0.50)
- **mAP50-95** (Mean Average Precision across IoU thresholds 0.50 to 0.95)
- **Confusion Matrix** (To track inter-class misclassifications, e.g., Front vs Back)
- **Per-Class AP**
- **Per-Class Precision**
- **Per-Class Recall**

## Critical Evaluation Constraint
> [!IMPORTANT]
> **Overall mAP is NOT sufficient.**

Because `Extra_Areas` represents 55.85% of all annotations, the overall mAP will be heavily skewed by the model's ability to detect `Extra_Areas`. A high overall mAP could completely mask catastrophic failure on the core structural panels.

The following minority classes:
- `Front_Panel`
- `Back_Panel`
- `Top_Panel`
- `Bottom_Part`

**MUST** be evaluated independently. A model is only successful if it achieves high Recall and mAP50 specifically on these minority structural faces.
