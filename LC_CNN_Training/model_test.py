# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 22:38:47 2026

@author: tomla
"""

# predict_and_visualize.py
# Standalone script to load trained CNN, pick random sample from Samples/,
# run prediction, and visualize: input image grid, ground-truth director (mid-slice),
# predicted director (mid-slice).
# Assumptions: cnn_model.pth exists, Samples/ has .bmp and .vti pairs.
# Run: python predict_and_visualize.py --data_dir Samples --model_path cnn_model.pth

import os
import glob
import random
import argparse
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import nemaktis as nm  # For loading .vti

# --- U-Net from training script (copy-pasted for standalone) ---
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=48, out_ch=30, features=32):
        super().__init__()
        
        self.enc1 = DoubleConv(in_ch, features)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(features, features*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(features*2, features*4)
        
        self.bottleneck = DoubleConv(features*4, features*8)
        
        self.up3 = nn.ConvTranspose2d(features*8, features*4, 2, stride=2)
        self.dec3 = DoubleConv(features*8, features*4)
        self.up2 = nn.ConvTranspose2d(features*4, features*2, 2, stride=2)
        self.dec2 = DoubleConv(features*4, features*2)
        self.up1 = nn.ConvTranspose2d(features*2, features, 2, stride=2)
        self.dec1 = DoubleConv(features*2, features)
        
        self.final = nn.Conv2d(features, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        
        b = self.bottleneck(self.pool2(e3))
        
        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        
        return self.final(d1)

# --- Args ---
parser = argparse.ArgumentParser(description="Predict and Visualize Random Sample")
parser.add_argument("--data_dir", type=str, default="Samples", help="Path to Samples folder")
parser.add_argument("--model_path", type=str, default="cnn_model.pth", help="Path to trained model")
args = parser.parse_args()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model ---
model = UNet().to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()
print("Loaded model from", args.model_path)

# --- Pick Random Sample ---
image_files = glob.glob(os.path.join(args.data_dir, "*.bmp"))
if not image_files:
    raise ValueError(f"No .bmp files in {args.data_dir}")

random_path = random.choice(image_files)
id_base = os.path.basename(random_path).replace(".bmp", "")
print(f"Selected sample: {id_base}")

# Load image
img = np.array(Image.open(random_path))  # (160,160,3)

# Split into sub-images and stack (like in Dataset)
sub_images = []
for i in range(4):
    for j in range(4):
        sub = img[i*40:(i+1)*40, j*40:(j+1)*40, :]
        sub_images.append(sub)
input_np = np.concatenate(sub_images, axis=-1)  # (40,40,48)
input_np = input_np.transpose(2, 0, 1)  # (48,40,40)
input_tensor = torch.from_numpy(input_np).float().unsqueeze(0).to(device) / 255.0

# Load ground-truth director
vti_path = os.path.join(args.data_dir, f"{id_base}.vti")
nfield_gt = nm.DirectorField(vti_file=vti_path)
director_gt = nfield_gt.vals  # (10,40,40,3) Nz,Ny,Nx,vec

# --- Run Prediction ---
with torch.no_grad():
    pred_flat = model(input_tensor)  # (1,30,40,40)
pred = pred_flat.view(1, 30, 40, 40).permute(0,2,3,1).view(1,40,40,10,3)
pred = torch.nn.functional.normalize(pred, dim=-1)  # Unit vectors
director_pred = pred.squeeze(0).cpu().numpy()  # (40,40,10,3) Ny,Nx,Nz,vec

# --- Save Predicted .vti (optional, for Paraview) ---
nfield_pred = nm.DirectorField(mesh_lengths=(20,20,5), mesh_dimensions=(40,40,10))
nfield_pred.vals = director_pred.transpose(2,0,1,3)  # Back to (Nz,Ny,Nx,3)
nfield_pred.save_to_vti(f"predicted_{id_base}.vti")
print(f"Saved predicted .vti to predicted_{id_base}.vti")

# --- Visualization ---
# 1. Input Image Grid
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Input Polarized Image Grid")
plt.axis("off")

# Helper to visualize director slice (mid-z, quiver for direction)
def plot_director(ax, director, title):
    """
    director: shape (Ny, Nx, Nz, 3) or (40, 40, 10, 3)
    """
    # Make sure we have (Ny, Nx, Nz, 3)
    if director.ndim != 4 or director.shape[-1] != 3:
        raise ValueError(f"Expected (Ny, Nx, Nz, 3), got {director.shape}")

    # Take mid z-slice
    mid_slice = director[:, :, 5, :]          # shape (40, 40, 3)

    # Downsample for visualization (every 2nd or 3rd point)
    step = 4
    y, x = np.mgrid[0:40:step, 0:40:step]     # shape (20, 20)

    # Important: slice the vectors with THE SAME step
    ux = mid_slice[::step, ::step, 0]         # (20, 20)
    uy = mid_slice[::step, ::step, 1]         # (20, 20)

    # Check shapes match
    assert ux.shape == x.shape == y.shape, \
        f"Shape mismatch: positions {x.shape}, vectors {ux.shape}"

    ax.quiver(x, y, ux, uy,
              angles='xy',
              scale_units='xy',
              scale=1,
              color='b',
              width=0.005)

    ax.set_title(title)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # usually matches image coordinates
    ax.set_xlim(-1, 41)
    ax.set_ylim(-1, 41)

# 2. Ground-Truth Director (mid-z slice)
plt.subplot(1, 3, 2)
plot_director(plt.gca(), director_gt.transpose(1,2,0,3), "Ground-Truth Director (Mid-Z)")

# 3. Predicted Director (mid-z slice)
plt.subplot(1, 3, 3)
plot_director(plt.gca(), director_pred, "Predicted Director (Mid-Z)")

plt.tight_layout()
plt.show()

print("Visualization complete. For full 3D view, open .vti files in Paraview.")