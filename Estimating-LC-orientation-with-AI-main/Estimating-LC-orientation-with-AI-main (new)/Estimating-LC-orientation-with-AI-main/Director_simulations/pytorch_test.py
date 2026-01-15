# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 23:14:20 2026

@author: user
"""

import torch
import time
print("PyTorch version:", torch.__version__)
print("CUDA available?:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

size = 8192

a = torch.randn(size, size)
b = torch.randn(size, size)

# CPU
start = time.time()
c = torch.matmul(a, b)
cpu_time = time.time() - start
print(f"CPU ({size}×{size}): {cpu_time:.4f} s")

# GPU
if torch.cuda.is_available():
    a = a.cuda()
    b = b.cuda()
    
    # Warm-up
    _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    print(f"GPU ({size}×{size}): {gpu_time:.4f} s")
    print(f"→ GPU is {cpu_time/gpu_time:.1f}× FASTER")
else:
    print("No CUDA detected")
    