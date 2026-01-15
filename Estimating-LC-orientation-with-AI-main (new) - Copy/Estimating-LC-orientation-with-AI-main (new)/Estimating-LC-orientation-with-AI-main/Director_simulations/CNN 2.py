# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 18:48:50 2026

@author: user
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import os

# ─── 1. Device (GPU if available) ───────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")               # Should print: cuda
print(f"GPU: {torch.cuda.get_device_name(0)}") # Your GTX 1650 Max-Q

# ─── 2. Data ─────────────────────────────────────────────────────────────
class SingleFolderDataset(Dataset):
    """Load all images from one folder, no labels (for inference/testing)"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all image files (supports common extensions)
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        self.image_paths = [
            os.path.join(root_dir, f) for f in os.listdir(root_dir)
            if os.path.isfile(os.path.join(root_dir, f)) and f.lower().endswith(valid_extensions)
        ]
        print(f"Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # force RGB
        
        if self.transform:
            image = self.transform(image)
            
        # Return dummy label (or None) since no classes
        return image, 0  # or return image, None if you don't need label
    
# Example: using CIFAR-10 (change to your own liquid crystal dataset)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # good for most images
])

your_folder = "C:/Users/tomla/Downloads/Estimating-LC-orientation-with-AI-main (new)/Estimating-LC-orientation-with-AI-main (new)/Estimating-LC-orientation-with-AI-main/Director_simulations/Samples"

# Full dataset (replace with your own folder/dataset)
full_dataset = SingleFolderDataset(root_dir=your_folder, transform=transform)

# Split train / validation
train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

# DataLoaders – important for GTX 1650: small batch + pin_memory
BATCH_SIZE = 32     # ← start here!  Try 16/64 if OOM or too slow
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=True)   # pin_memory → faster CPU→GPU copy

val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False,
                        num_workers=0, pin_memory=True)

# ─── 3. Simple CNN Model ─────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):  # change num_classes for your problem
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc1 = nn.Linear(64 * 4 * 4, 128)   # after two pools: 32→16→8 for 32×32 input
        self.fc2 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)           # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = SimpleCNN(num_classes=10).to(device)   # ← VERY IMPORTANT: move to GPU!

# ─── 4. Loss & Optimizer ─────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()              # most common for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # or SGD with momentum
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # optional

# ─── 5. Training Loop ────────────────────────────────────────────────────
NUM_EPOCHS = 20   # start small, increase later

for epoch in range(NUM_EPOCHS):
    model.train()                # important!
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(train_loader):
        # ─── Most important lines for GPU ───────────────────────────────
        images = images.to(device)    # move batch to GPU
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if i % 100 == 99:    # print every 100 batches
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                  f'Step [{i+1}/{len(train_loader)}], '
                  f'Loss: {running_loss/100:.4f}, '
                  f'Accuracy: {100*correct/total:.2f}%')
            running_loss = 0.0
            correct = total = 0
    
    scheduler.step()  # optional
    
    # Quick validation (optional but recommended)
    model.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    print(f'Epoch [{epoch+1}] Validation Accuracy: {100*val_correct/val_total:.2f}%')

print("Training finished!")