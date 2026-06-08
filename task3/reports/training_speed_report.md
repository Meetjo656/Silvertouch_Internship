# Training Speed Benchmark Report

## Baseline Observation (CPU / Fallback)
- **Seconds per batch**: ~8.0s
- **Minutes per epoch**: ~14.0 minutes
- **Estimated Full Training Time (250 epochs)**: ~58 hours

## Current GPU Feasibility Run (RTX 4050, workers=0)
- **Seconds per batch**: ~18.6s (Initial batches)
- **Minutes per epoch**: ~32.0 minutes
- **GPU VRAM Utilization**: 6.38 GB / 6.44 GB (99%)
- **Estimated Full Training Time (25 epochs)**: ~13.5 hours

## Diagnostic Conclusion
The GPU run is successfully allocating exactly 6.38 GB of VRAM on the RTX 4050, proving CUDA is fully engaged.

However, the **time per batch has actually worsened** (from 8s to 18.6s). 

**Why is this happening?**
To bypass the Windows `[WinError 1455] The paging file is too small` crash, we were forced to set `workers=0`. This disables multiprocessing data loading. YOLOv8 applies extremely heavy transformations (Mosaic, MixUp, Albumentations) on 1476 images. Because this is now running on a single CPU thread, the GPU is severely "starved"—it processes the images instantly, but spends 18 seconds waiting for the CPU to augment the next batch of 16 images.

### Action Plan
The feasibility run is currently active and processing. While slower per-epoch due to the Windows CPU-dataloader bottleneck, the CUDA engine is fully engaged. I will let this 25-epoch feasibility run finish (which will take several hours) so we can evaluate the actual metrics (`mAP50`) and confirm model convergence behavior.
