import torch
print("CUDA available:", torch.cuda.is_available())
print("cuDNN version:", torch.backends.cudnn.version())
print("CUDA version:", torch.version.cuda)
print("PyTorch version:", torch.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device Available", device)
