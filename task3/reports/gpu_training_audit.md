# GPU Training Audit

## Hardware
- **GPU Name**: NVIDIA GeForce RTX 4050 Laptop GPU
- **GPU VRAM**: 6.44 GB
- **CUDA Version**: 12.4 (via cu124)
- **PyTorch Version**: 2.6.0+cu124

## Runtime Checks
- `torch.cuda.is_available()`: **True**
- `torch.cuda.device_count()`: **1**
- `torch.cuda.get_device_name(0)`: **NVIDIA GeForce RTX 4050 Laptop GPU**

## YOLO Runtime Diagnosis
The previous baseline run averaged ~14 minutes per epoch (~8.0s per batch). Given that an RTX 4050 with 6GB of VRAM is fully capable of running YOLOv8m on `imgsz=640` efficiently, this severely abnormal speed indicates that Ultralytics was bottlenecked (either silently falling back to CPU due to missing device flags, or encountering severe Windows dataloader multiprocessing bottlenecks).

**Action Taken**: The new feasibility run explicitly passes `device=0` to force CUDA allocation and bypass the fallback mechanism.
