---
language: en
tags:
- object-detection
- packaging
- computer-vision
- rf-detr
---

# Model Card: Packaging Segmentation RF-DETR

## Model Details
- **Architecture:** RF-DETR Small (Roboflow)
- **Task:** Object Detection & Component Segmentation
- **Input Size:** 640x640 (Strict)
- **Framework:** PyTorch & rfdetr

## Purpose
Designed to identify and segment bounding boxes of flat packaging designs into their semantic structural components (Front Panel, Side Panel, etc.). This serves as the primary gating mechanism before downstream OCR and Rx extraction pipelines.

## Dataset & Classes
Trained on a proprietary dataset annotated in COCO format.

**8-Class Ontology:**
0. Artwork
1. not_artwork
2. Front_Panel
3. Side_Panel
4. Back_Panel
5. Extra_Areas
6. Bottom_Panel
7. Top_Panel

## Training Configuration
(Refer to the specific `experiment.yaml` in the model output directory for exact parameters).
- Backbone: DINOv2 (via RF-DETR Small)
- Optimization: AdamW
- Typical Confidence Threshold: 0.25

## Evaluation & Benchmarks
Pending execution of `benchmark.py` and `evaluate.py`.
(Fill this section after quantitative evaluation is complete with mAP, F1, and Inference Latency).

## Limitations & Known Failure Modes
- **Extreme Aspect Ratios:** Models occasionally struggle on extremely long or tall packaging (e.g., tube boxes).
- **Domain Shift:** Sensitive to synthetic backgrounds or vastly different packaging materials not present in the training distribution.
- **Duplicate Detections:** Highly dense clusters might trigger multiple bounding boxes if NMS/IoU thresholds are not tuned correctly.

## Ethical Considerations & Usage
This model is intended strictly for internal enterprise automation of packaging validation.

## Authors
Created by Silvertouch Internship Team.
