# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 21:37:06 2026

@author: tomla
"""

# train_cnn.py
# Script to train a U-Net CNN for predicting 3D director fields from polarized micrograph grids.
# Assumptions:
# - Data in "Samples/" folder: {ID}.bmp (160x160x3 RGB grid of 4x4=16 sub-images)
# - {ID}.vti (director field, shape (Nz=10, Ny=40, Nx=40, 3))
# - PyTorch and nemaktis installed (pip install torch nemaktis)
# - Run: python train_cnn.py --data_dir Samples --epochs 50 --batch_size 8

import os
import glob
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import nemaktis as nm  # For loading .vti files

# --- Hyperparameters (configurable) ---
parser = argparse.ArgumentParser(description="Train CNN for LC Director Prediction")
parser.add_argument("--data_dir", type=str, default="Samples", help="Path to Samples folder")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
parser.add_argument("--save_path", type=str, default="cnn_model.pth", help="Model save path")
args = parser.parse_args()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Dataset Class ---
class LCDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = sorted(glob.glob(os.path.join(data_dir, "*.bmp")))
        self.num_sub_images = 16  # 4x4 grid
        self.sub_img_size = 40    # Each sub-image is 40x40 (matching Ny,Nx)
        self.channels_in = self.num_sub_images * 3  # 16 RGB → 48 channels
        self.z_depth = 10         # Nz=10
        self.channels_out = self.z_depth * 3  # 10 layers * 3 components = 30 channels
        
        if not self.image_files:
            raise ValueError(f"No .bmp files found in {data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image (.bmp)
        img_path = self.image_files[idx]
        img = np.array(Image.open(img_path))  # (160,160,3)
        
        # Split into 4x4 grid of 40x40x3 sub-images
        sub_images = []
        for i in range(4):
            for j in range(4):
                sub = img[i*40:(i+1)*40, j*40:(j+1)*40, :]
                sub_images.append(sub)
        
        # Stack along channel dim: (40,40,48)
        input_tensor = np.concatenate(sub_images, axis=-1)
        input_tensor = input_tensor.transpose(2, 0, 1)  # (48,40,40)
        input_tensor = torch.from_numpy(input_tensor).float() / 255.0  # Normalize [0,1]

        # Load director field (.vti)
        id_base = os.path.basename(img_path).replace(".bmp", "")
        vti_path = os.path.join(self.data_dir, f"{id_base}.vti")
        if not os.path.exists(vti_path):
            raise FileNotFoundError(f"Missing {vti_path}")
        
        nfield = nm.DirectorField(vti_file=vti_path)
        director = nfield.vals  # (10,40,40,3) Nz,Ny,Nx,vec
        
        # Flatten z and vec: (40,40,30) Ny,Nx, (Nz*3)
        director = director.transpose(1,2,0,3)  # (40,40,10,3)
        director_flat = director.reshape(40, 40, -1)  # (40,40,30)
        director_flat = director_flat.transpose(2, 0, 1)  # (30,40,40)
        
        label_tensor = torch.from_numpy(director_flat).float()

        return input_tensor, label_tensor

# --- U-Net Architecture (Simple version for 40x40 images) ---
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
        
        # Encoder
        self.enc1 = DoubleConv(in_ch, features)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(features, features*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(features*2, features*4)
        
        # Bottleneck
        self.bottleneck = DoubleConv(features*4, features*8)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(features*8, features*4, 2, stride=2)
        self.dec3 = DoubleConv(features*8, features*4)
        self.up2 = nn.ConvTranspose2d(features*4, features*2, 2, stride=2)
        self.dec2 = DoubleConv(features*4, features*2)
        self.up1 = nn.ConvTranspose2d(features*2, features, 2, stride=2)
        self.dec1 = DoubleConv(features*2, features)
        
        # Output
        self.final = nn.Conv2d(features, out_ch, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        
        # Bottleneck
        b = self.bottleneck(self.pool2(e3))  # Extra pool? Wait, adjust for 40x40
        
        # Note: 40x40 → after 2 pools: 10x10, adjust if needed
        
        # Decoder
        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))  # Skip connection
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        
        return self.final(d1)

# --- Custom Loss (MSE + Norm Regularization) ---
def director_loss(pred, target):
    mse = nn.MSELoss()(pred, target)
    
    # Encourage unit norm: reshape to (B,40,40,10,3), compute norm
    B = pred.shape[0]
    pred_reshaped = pred.view(B, 30, 40, 40).permute(0,2,3,1).view(B,40,40,10,3)
    norms = torch.norm(pred_reshaped, dim=-1)  # (B,40,40,10)
    norm_loss = torch.mean((norms - 1.0)**2)
    
    return mse + 0.1 * norm_loss

# --- Training Function ---
def train_model(model, train_loader, val_loader, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = director_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += director_loss(outputs, labels).item()
        
        val_loss /= len(val_loader)
        # scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved best model at epoch {epoch+1}")

# --- Main ---
dataset = LCDataset(args.data_dir)
print(f"Loaded {len(dataset)} samples")

# Split train/val
val_size = int(len(dataset) * args.val_split)
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

# Model
model = UNet().to(device)

# Train
train_model(model, train_loader, val_loader, args.epochs, args.lr)

print("Training complete. Model saved to", args.save_path)