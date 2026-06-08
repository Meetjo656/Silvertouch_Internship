# Class Balance Audit

## Distribution Breakdown

| Class | Count | Percentage |
| :--- | ---: | ---: |
| Front_Panel | 217 | 7.54% |
| Side_Panel | 434 | 15.07% |
| Back_Panel | 197 | 6.84% |
| Extra_Areas | 1608 | 55.85% |
| Bottom_Part | 213 | 7.40% |
| Top_Panel | 210 | 7.29% |

## Summary Statistics

- **Largest Class**: `Extra_Areas` (1608 instances)
- **Smallest Class**: `Back_Panel` (197 instances)
- **Largest / Smallest Ratio**: 8.16

## Conclusion
Based on an 8.16x disparity between the majority and minority classes, the dataset is:
**Moderately Imbalanced**

The model may naturally bias its anchor predictions toward `Extra_Areas` without appropriate focal loss or class weighting during training.
