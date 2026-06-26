import torch

print(f"PyTorch Version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    print(f"Device Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i} Name: {torch.cuda.get_device_name(i)}")
        # Get memory info
        prop = torch.cuda.get_device_properties(i)
        print(f"Device {i} VRAM: {prop.total_memory / 1e9:.2f} GB")
else:
    print("No CUDA devices found.")
