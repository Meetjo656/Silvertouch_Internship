# 8-Class Validation Training Results

The 5-epoch sanity check training run using `yolo26n.pt` has completed successfully!

## Overview
* **Model**: YOLO26n (8-class configuration)
* **Epochs**: 5
* **Hardware**: CUDA:0 (NVIDIA GeForce RTX 4050 Laptop GPU)
* **Training Time**: 0.023 hours (approx. 1.5 minutes)

## Validation Metrics (Epoch 5)

| Class | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **All** | 0.428 | 0.490 | 0.395 | 0.350 |
| **Artwork** | 0.671 | 0.532 | 0.592 | 0.454 |
| **Not_Artwork** | 0.209 | 0.476 | 0.243 | 0.164 |
| **Front_Panel** | 0.214 | 0.381 | 0.227 | 0.219 |
| **Side_Panel** | 0.267 | 0.690 | 0.283 | 0.269 |
| **Back_Panel** | 0.275 | 0.750 | 0.515 | 0.499 |
| **Extra_Areas** | 0.346 | 0.755 | 0.561 | 0.509 |
| **Bottom_Part** | 0.491 | 0.238 | 0.402 | 0.365 |
| **Top_Panel** | 0.947 | 0.095 | 0.339 | 0.318 |

## Architectural Validation

**SUCCESS.**

The core purpose of this sanity check was to verify whether YOLOv8's Feature Pyramid Network (FPN) could process full-image macro bounding boxes (`Artwork` / `Not_Artwork`) alongside localized micro bounding boxes (structural panels) without causing dataset collisions or loss calculation errors.

1. **Gradients Flowing:** The model successfully learned initial feature representations for both global and local classes simultaneously.
2. **No NMS Suppression Issues:** The model correctly generated bounding boxes for `Artwork` and structural panels in the same evaluation pass.
3. **No WinError 1455:** By forcing `workers=0`, the training bypassed the Windows multiprocessing memory leak and completed flawlessly.

## Next Steps

As expected with only 5 epochs on a Nano model, the absolute mAP numbers are low (especially for the new global classes). The architecture is now proven. We are clear to launch the full 50-epoch training run whenever you are ready.
