import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pyvista as pv
import numpy as np


class LiquidCrystalPairDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        data_dir: folder that contains 1.bmp, 1.vti, 2.bmp, 2.vti, ...
        """
        self.data_dir = data_dir
        
        # Find all .bmp files and extract their numbers
        bmp_files = [f for f in os.listdir(data_dir) if f.endswith('.bmp')]
        self.indices = sorted(
            [int(f.split('.')[0]) for f in bmp_files if f.split('.')[0].isdigit()]
        )
        
        if not self.indices:
            raise ValueError("No .bmp files found in the directory.")
        
        print(f"Found {len(self.indices)} samples (from {min(self.indices)} to {max(self.indices)})")
        
        # Optional: image preprocessing
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),                  # [H,W] or [H,W,3] → [C,H,W]
            transforms.Normalize(mean=[0.5], std=[0.5]),  # good starting point for grayscale
            # transforms.Resize((256, 256)),        # uncomment & adjust if needed
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        number = self.indices[idx]
        
        # File paths
        img_path = os.path.join(self.data_dir, f"{number}.bmp")
        vti_path = os.path.join(self.data_dir, f"{number}.vti")
        
        # Load image
        image = Image.open(img_path).convert('L')   # 'L' = grayscale
        image = self.transform(image)               # → tensor ~ [1, H, W]
        
        # Load VTI (director field)
        mesh = pv.read(vti_path)
        
        # IMPORTANT: you need to know the exact name of the array
        # Common names: 'director', 'directors', 'n', 'orientation', 'vector'
        # Run this once to check:
        # print(mesh.point_data.keys())
        
        array_name = 'directors'                    # ← change this to the correct name!
        if array_name not in mesh.point_data:
            raise KeyError(f"Array '{array_name}' not found. Available: {list(mesh.point_data.keys())}")
        
        directors = mesh.point_data[array_name]     # usually shape (N_points, 3)
        
        # Reshape to image-like grid
        # You need to know the grid size — usually stored in mesh.dimensions
        dims = mesh.dimensions                      # (nx, ny, nz)
        
        # Most liquid crystal 2D simulations are 2D → nz == 1
        if dims[2] == 1:
            h, w = dims[1], dims[0]
            directors = directors.reshape((h, w, 3))
        else:
            # 3D case — you'll need to decide how to handle (slice? flatten?)
            # For now assuming 2D
            raise ValueError(f"3D grid detected ({dims}). Please handle 3D case separately.")
        
        # Convert to torch tensor, shape [3, H, W]
        directors = torch.from_numpy(directors.astype(np.float32)).permute(2, 0, 1)
        
        return image, directors


# ────────────────────────────────────────────────
# Usage example
# ────────────────────────────────────────────────

data_dir = "path/to/your/folder"   # ← change this

dataset = LiquidCrystalPairDataset(data_dir)

# Quick test
img, dir_field = dataset[0]
print("Image shape:", img.shape)          # should be [1, H, W]
print("Director field shape:", dir_field.shape)  # should be [3, H, W]

# Optional: visualize first sample quickly
import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.imshow(img[0], cmap='gray')
plt.title("Microscope image")
plt.subplot(1,2,2)
plt.quiver(dir_field[0][::4,::4], dir_field[1][::4,::4], scale=30)
plt.title("Director field (x,y components)")
plt.show()

# Then create dataloader as usual
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
