import torch

free, total = torch.cuda.mem_get_info()
print(f"CUDA memory: free={free/1e9:.2f} GB, total={total/1e9:.2f} GB")
