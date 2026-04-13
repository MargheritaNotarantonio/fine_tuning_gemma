import torch
print(f"Versione Torch: {torch.__version__}")
print(f"CUDA disponibile? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")