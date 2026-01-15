# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 03:35:57 2026

@author: tomlai
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
import re  # Added for parsing filenames

# ─── 1. Device (GPU if available) ───────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")               # Should print: cuda
print(f"GPU: {torch.cuda.get_device_name(0)}") # Your GTX 1650 Max-Q

# ─── 2. Data ─────────────────────────────────────────────────────────────
class SingleFolderDataset(Dataset):
    """Load all images from one folder, parse labels from filenames for regression"""
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
            
        # Parse label from filename (assumes format like "img_45.0.png" or "orientation_30.jpg" 
        # where the number is the angle; adjust regex if needed)
        filename = os.path.basename(img_path)
        match = re.search(r'(\d+\.?\d*)', filename)  # Find first float-like number
        if match:
            label = float(match.group(1))
        else:
            raise ValueError(f"Could not parse label from filename: {filename}")
        
        return image, label  # Label is now float (e.g., angle)
    
# Example: add augmentation for better generalization (especially for orientations)
transform = transforms.Compose([
    transforms.RandomRotation(180),  # Helpful for crystal symmetries
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # good for most images
])

your_folder = "C:/Users/user/Downloads/Estimating-LC-orientation-with-AI-main/Estimating-LC-orientation-with-AI-main (new)/Estimating-LC-orientation-with-AI-main/Director_simulations/Samples"

# Full dataset
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

# ─── 3. Simple CNN Model (changed to regression) ─────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc1 = nn.Linear(64 * 4 * 4, 128)   # after two pools
        self.fc2 = nn.Linear(128, 1)            # 1 output for regression (e.g., angle)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)           # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze()  # Remove extra dim for scalar output

model = SimpleCNN().to(device)   # ← VERY IMPORTANT: move to GPU!

# ─── 4. Loss & Optimizer (changed to MSE for regression) ────────────────
criterion = nn.MSELoss()               # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)  # or SGD with momentum
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # optional

# ─── 5. Training Loop (updated for regression: use MAE instead of accuracy) ───
NUM_EPOCHS = 20   # start small, increase later

for epoch in range(NUM_EPOCHS):
    model.train()                # important!
    running_loss = 0.0
    running_mae = 0.0
    total = 0
    
    for i, (images, labels) in enumerate(train_loader):
        # ─── Most important lines for GPU ───────────────────────────────
        images = images.to(device)    # move batch to GPU
        labels = labels.to(device).float()  # Ensure labels are float
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * labels.size(0)  # Accumulate total loss
        running_mae += torch.abs(outputs - labels).sum().item()  # Accumulate absolute errors
        total += labels.size(0)
        
        if i % 100 == 99:    # print every 100 batches
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                  f'Step [{i+1}/{len(train_loader)}], '
                  f'Loss: {running_loss/total:.4f}, '
                  f'MAE: {running_mae/total:.2f}')
            running_loss = 0.0
            running_mae = 0.0
            total = 0
    
    scheduler.step()  # optional
    
    # Quick validation (use MAE for regression)
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            outputs = model(images)
            val_loss += criterion(outputs, labels).item() * labels.size(0)
            val_mae += torch.abs(outputs - labels).sum().item()
            val_total += labels.size(0)
    
    val_loss /= val_total
    val_mae /= val_total
    print(f'Epoch [{epoch+1}] Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.2f}')

print("Training finished!")

# Optional: Save the model
torch.save(model.state_dict(), 'lc_orientation_model.pth')
print("Model saved to 'lc_orientation_model.pth'")