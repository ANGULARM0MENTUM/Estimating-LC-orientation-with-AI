# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 23:15:37 2026

@author: user
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import torchvision
from torchvision import datasets, transforms

import multiprocessing
multiprocessing.freeze_support()



import time
from datetime import datetime

# ─── 1. Configuration ───────────────────────────────────────────────────────
NUM_EPOCHS = 50
BATCH_SIZE = 32           # Start small for 4GB VRAM, increase to 64 if possible
LEARNING_RATE = 0.001
PATIENCE = 8              # For early stopping
IMAGE_SIZE = 128          # Adjust to your actual image size
NUM_CLASSES = 10          # ← CHANGE THIS to your number of director classes/configurations

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Enable some performance optimizations
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# ─── 2. Strong Data Augmentation (very important for LC textures) ────────────
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=180),           # LC patterns are often symmetric
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),  # microscope noise
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ─── 3. Load your dataset ────────────────────────────────────────────────────
# Example using folder structure: root/class1/, root/class2/, etc.
# CHANGE PATHS to your actual data location!
data_dir = "C:/Users/tomla/Downloads/Estimating-LC-orientation-with-AI-main (new)/Estimating-LC-orientation-with-AI-main (new)/Estimating-LC-orientation-with-AI-main/Director_simulations/Samples"   # ← your folder

full_dataset = datasets.ImageFolder(
    root=data_dir,
    transform=train_transform
)

# Split: 80% train, 20% validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Override val transform (no augmentation for validation)
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,                # 0 = safest & often fastest on Windows laptop
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

# ─── 4. Simple but effective CNN model ───────────────────────────────────────
class LC_CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMAGE_SIZE//8) * (IMAGE_SIZE//8), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = LC_CNN(num_classes=NUM_CLASSES).to(DEVICE)

# ─── 5. Loss, Optimizer, Scheduler, Scaler ───────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, verbose=True)
scaler = GradScaler()  # for mixed precision

# ─── 6. Training + Validation Loop ───────────────────────────────────────────
best_val_acc = 0.0
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    
    # ─── Training ─────────────────────────────────────────────────────────────
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_loss /= len(train_loader)
    train_acc = 100 * train_correct / train_total
    
    # ─── Validation ───────────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100 * val_correct / val_total
    
    # ─── Scheduler & Progress ─────────────────────────────────────────────────
    scheduler.step(val_acc)  # Reduce LR based on validation accuracy
    
    epoch_time = time.time() - start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
          f"Time: {epoch_time:.1f}s")
    
    # ─── Save best model & Early Stopping ─────────────────────────────────────
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_lc_director_model.pth")
        print(f"→ New best validation accuracy: {best_val_acc:.2f}% - model saved!")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

print("Training finished!")
print(f"Best validation accuracy achieved: {best_val_acc:.2f}%")