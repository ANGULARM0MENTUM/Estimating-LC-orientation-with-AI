# -*- coding: utf-8 -*-
"""
3D Director Field Training  — Z-Depth-Aware Version
=====================================================
Key improvements over v1:
  1. Sinusoidal Z positional encoding  — every voxel knows its depth
  2. Deeper ZExpansionBridge           — 3 stacked 3D conv layers so the
                                         network can learn Z-varying patterns
  3. Z-aware skip fusion               — positional encoding injected into
                                         each decoder skip-cat before fusion
  4. Z-weighted loss                   — harder interpolated mid-Z slices
                                         receive 2× the loss weight of the
                                         uniform top/bottom slices
  5. Larger z_seed (10 instead of 5)   — decoder has more Z resolution
                                         before the final trilinear stretch

Tensor shapes (MESH_DIMENSIONS=(80,80,20), IMG_SIZE=80, N_Z_SLICES=20):
  Input  : (B, 16,  80, 80)
  Output : (B,  3,  20, 80, 80)   (B, component, Z, Y, X)
  Target : (B,  3,  20, 80, 80)   loaded directly from .vti, no Z-averaging
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pyvista as pv
from pathlib import Path
from PIL import Image
import json
import random
from tqdm import tqdm
from scipy.ndimage import zoom

# ========================= CONFIG =========================
SAMPLES_DIR     = Path("Samples")
METADATA_JSON   = SAMPLES_DIR / "Samples_metadata.json"
MODEL_SAVE_PATH = SAMPLES_DIR / "director_model_full3d.pth"

IMG_SIZE        = 80
N_Z_SLICES      = 20
Z_SEED          = 10   # ↑ from 5: more Z resolution throughout the decoder
BASE_CHANNELS   = 32
BATCH_SIZE      = 8
EPOCHS          = 200
INITIAL_LR      = 3e-4
TRAIN_SPLIT     = 0.85
PATIENCE        = 25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ========================= DATA LOADING =========================

def load_full_composite(sample_id: int) -> np.ndarray:
    """
    Load the 4×4 POM composite and return (16, H, W) float32 in [0,1].
    Each of the 16 tiles corresponds to a (polariser, analyser) angle pair.
    """
    bmp_path = SAMPLES_DIR / f"{sample_id}.bmp"
    img = np.array(Image.open(bmp_path))
    h, w = img.shape[:2]
    th, tw = h // 4, w // 4
    channels = []
    for row in range(4):
        for col in range(4):
            tile = img[row*th:(row+1)*th, col*tw:(col+1)*tw]
            gray = np.mean(tile, axis=2) if tile.ndim == 3 else tile
            channels.append(gray.astype(np.float32) / 255.0)
    return np.stack(channels, axis=0)   # (16, H, W)


def load_full_3d_director(vti_path: Path,
                           target_xy: int = IMG_SIZE,
                           target_z:  int = N_Z_SLICES) -> np.ndarray:
    """
    Load the complete 3D director field from .vti without Z-averaging.
    Returns (3, Z, Y, X) float32 unit-normalised.
    """
    grid = pv.read(str(vti_path))
    vec  = grid.point_data['n']           # (Nx*Ny*Nz, 3)
    dims = grid.dimensions                # (Nx, Ny, Nz)
    n3d  = vec.reshape((dims[2], dims[1], dims[0], 3)).astype(np.float32)

    src_z, src_y, src_x = n3d.shape[:3]
    if (src_z, src_y, src_x) != (target_z, target_xy, target_xy):
        n3d = zoom(n3d,
                   (target_z/src_z, target_xy/src_y, target_xy/src_x, 1),
                   order=1)

    norm = np.sqrt(np.sum(n3d**2, axis=-1, keepdims=True) + 1e-10)
    n3d  = n3d / norm
    return np.transpose(n3d, (3, 0, 1, 2)).astype(np.float32)  # (3,Z,Y,X)


# ========================= DATASET =========================

class LC_3D_Dataset(Dataset):
    def __init__(self, sample_ids):
        self.sample_ids = sample_ids

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        x = torch.from_numpy(load_full_composite(sid)).float()
        y = torch.from_numpy(
                load_full_3d_director(SAMPLES_DIR / f"{sid}.vti")
            ).float()
        return x, y


# ========================= POSITIONAL ENCODING =========================

def make_z_encoding(z: int, channels: int, device) -> torch.Tensor:
    """
    Sinusoidal positional encoding along the Z axis.

    Returns a (1, channels, z, 1, 1) tensor that can be broadcast-added
    to any (B, channels, z, H, W) feature volume.

    Uses the standard transformer sin/cos encoding:
        PE[2i]   = sin(pos / 10000^(2i/channels))
        PE[2i+1] = cos(pos / 10000^(2i/channels))

    Why sinusoidal?
    - Smooth, continuous representation of depth position
    - Generalises to unseen Z values if mesh depth ever changes
    - Alternating sin/cos means the model can learn any linear combination
      of depth-dependent patterns (e.g. linear interpolation between top/bottom)
    """
    pe       = torch.zeros(channels, z)          # (C, Z)
    position = torch.arange(z, dtype=torch.float).unsqueeze(0)  # (1, Z)
    div_term = torch.exp(
        torch.arange(0, channels, 2, dtype=torch.float)
        * (-np.log(10000.0) / channels)
    )                                            # (C/2,)
    pe[0::2, :] = torch.sin(position * div_term.unsqueeze(1))
    pe[1::2, :] = torch.cos(position * div_term.unsqueeze(1))
    # Reshape to (1, C, Z, 1, 1) for broadcasting
    return pe.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)


# ========================= MODEL =========================

class Encoder2D(nn.Module):
    """
    2D U-Net encoder on the 16-channel POM composite.
    Outputs multi-scale skip features at 4 resolutions.
    """
    def __init__(self, base=32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = self._block(16,     base)       # → (B, base,   80, 80)
        self.enc2 = self._block(base,   base*2)     # → (B, base*2, 40, 40)
        self.enc3 = self._block(base*2, base*4)     # → (B, base*4, 20, 20)
        self.enc4 = self._block(base*4, base*8)     # → (B, base*8, 10, 10)

    @staticmethod
    def _block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin,  cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        return e1, e2, e3, e4


class ZExpansionBridge(nn.Module):
    """
    Converts 2D bottleneck (B, cin, H, W) → 3D volume (B, cout, z_seed, H, W).

    Improvements vs v1:
    - Sinusoidal Z positional encoding added after tiling, before any 3D conv.
      This gives the network an explicit depth signal from the very start.
    - Three stacked 3D conv layers (vs one before) so the network has enough
      capacity to learn how POM features translate into depth-varying structure.
    """
    def __init__(self, cin, cout, z_seed=10):
        super().__init__()
        self.z_seed = z_seed
        self.cout   = cout

        self.project = nn.Sequential(
            nn.Conv2d(cin, cout, 1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )
        # 3 stacked 3D convs — enough depth to learn Z-varying patterns
        self.refine = nn.Sequential(
            nn.Conv3d(cout, cout, (3,3,3), padding=1),
            nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, (3,3,3), padding=1),
            nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, (3,3,3), padding=1),
            nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x  = self.project(x)                                         # (B, cout, H, W)
        x  = x.unsqueeze(2).expand(-1, -1, self.z_seed, -1, -1).contiguous()
                                                                     # (B, cout, z_seed, H, W)
        # Inject Z positional encoding before 3D convs
        pe = make_z_encoding(self.z_seed, self.cout, x.device)       # (1, cout, z_seed, 1, 1)
        x  = x + pe
        x  = self.refine(x)
        return x                                                     # (B, cout, z_seed, H, W)


class Decoder3D(nn.Module):
    """
    3D decoder: (B, base*8, z_seed, H/8, W/8) → (B, 3, Z_target, H, W)

    Improvements vs v1:
    - Z positional encoding is also injected at each skip-fusion stage so
      the high-resolution 2D skips become Z-aware when fused with 3D features.
    - Final Z stretch uses learned 3D conv refinement after trilinear upsample
      (instead of raw trilinear alone) so the model can correct interpolation
      artefacts introduced by the stretch.
    """
    def __init__(self, base=32, z_seed=10, z_target=20):
        super().__init__()
        self.z_target = z_target
        self.base     = base

        # XY ×2 at each stage; Z unchanged (scale_factor=(1,2,2))
        self.up4 = self._upblock(base*8, base*4)
        self.up3 = self._upblock(base*4, base*2)
        self.up2 = self._upblock(base*2, base)

        # Fusion: 3D feature (C) + Z-aware broadcast skip (C) → C
        self.fuse4 = self._fuse3d(base*4 + base*4, base*4)
        self.fuse3 = self._fuse3d(base*2 + base*2, base*2)
        self.fuse2 = self._fuse3d(base   + base,   base)

        # After trilinear Z-stretch: refine to remove interpolation artefacts
        self.z_refine = nn.Sequential(
            nn.Conv3d(base, base, (3,3,3), padding=1),
            nn.BatchNorm3d(base), nn.ReLU(inplace=True),
        )

        self.final = nn.Conv3d(base, 3, 1)

    @staticmethod
    def _upblock(cin, cout):
        return nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
            nn.Conv3d(cin, cout, (1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
        )

    @staticmethod
    def _fuse3d(cin, cout):
        return nn.Sequential(
            nn.Conv3d(cin, cout, (3,3,3), padding=1),
            nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
        )

    def _z_aware_skip(self, skip_2d: torch.Tensor, z: int, c: int) -> torch.Tensor:
        """
        Broadcast a 2D skip (B,C,H,W) to 3D (B,C,Z,H,W) and add
        sinusoidal Z positional encoding so the skip features know their
        depth when fused with the 3D decoder features.
        """
        skip_3d = skip_2d.unsqueeze(2).expand(-1, -1, z, -1, -1).contiguous()
        pe      = make_z_encoding(z, c, skip_2d.device)  # (1,C,Z,1,1)
        return skip_3d + pe

    def forward(self, bottleneck, e1, e2, e3):
        # bottleneck: (B, base*8, z_seed, H/8, W/8)

        u = self.up4(bottleneck)
        u = self.fuse4(torch.cat(
            [u, self._z_aware_skip(e3, u.shape[2], self.base*4)], dim=1))

        u = self.up3(u)
        u = self.fuse3(torch.cat(
            [u, self._z_aware_skip(e2, u.shape[2], self.base*2)], dim=1))

        u = self.up2(u)
        u = self.fuse2(torch.cat(
            [u, self._z_aware_skip(e1, u.shape[2], self.base)], dim=1))

        # Stretch Z: z_seed → z_target with trilinear, then refine
        if u.shape[2] != self.z_target:
            u = F.interpolate(
                u,
                size=(self.z_target, u.shape[3], u.shape[4]),
                mode='trilinear', align_corners=False,
            )
        u = self.z_refine(u)   # learned correction of interpolation artefacts

        u = self.final(u)      # (B, 3, Z_target, H, W)

        # Per-voxel unit normalisation
        norm = torch.sqrt(torch.sum(u**2, dim=1, keepdim=True) + 1e-8)
        return u / norm


class Director3DNet(nn.Module):
    """
    Full model:  POM (B,16,H,W)  →  3D director field (B,3,Z,H,W)

      Encoder2D  →  ZExpansionBridge  →  Decoder3D
           ↘_____ Z-aware skip connections ________↗
    """
    def __init__(self, base=BASE_CHANNELS, z_seed=Z_SEED, z_target=N_Z_SLICES):
        super().__init__()
        self.encoder = Encoder2D(base=base)
        self.bridge  = ZExpansionBridge(cin=base*8, cout=base*8, z_seed=z_seed)
        self.decoder = Decoder3D(base=base, z_seed=z_seed, z_target=z_target)

    def forward(self, x):
        e1, e2, e3, e4 = self.encoder(x)
        bottleneck      = self.bridge(e4)
        return self.decoder(bottleneck, e1, e2, e3)


# ========================= LOSS =========================

def build_z_weight(z: int, device) -> torch.Tensor:
    """
    Per-Z-slice loss weight: cosine profile peaking at the centre (z=mid).

    Rationale: uniform top/bottom Z slices are easier to predict; the
    interpolated middle layers are harder and most informative for the 3D task.
    Centre slices get weight ~2×, boundary slices get weight ~1×.

    Shape: (1, 1, Z, 1, 1) — broadcast-compatible with (B,3,Z,H,W).
    """
    zf = torch.arange(z, dtype=torch.float) / (z - 1)          # [0, 1]
    # cos profile:  1 + cos(2π·(z-0.5)) / 2   →  range [0.5, 1.5] → shift to [1, 2]
    w  = 1.0 + torch.cos(2.0 * np.pi * (zf - 0.5)) * 0.5 + 0.5
    return w.view(1, 1, z, 1, 1).to(device)


def director_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred, target : (B, 3, Z, H, W)  unit vectors

    1. Headless angular loss (|cos|) — n ≡ -n symmetry
    2. Z-weighted: centre slices weighted 2×, top/bottom ~1×
    3. Unit-norm regulariser
    """
    cos = torch.sum(pred * target, dim=1)                        # (B, Z, H, W)

    # Headless angular error in degrees
    angle = torch.acos(cos.abs().clamp(0.0, 1.0 - 1e-7))        # (B, Z, H, W)
    angle_deg = angle * (180.0 / np.pi)

    # Z-weighted mean
    w          = build_z_weight(pred.shape[2], pred.device)      # (1,1,Z,1,1)
    angle_loss = (angle_deg * w.squeeze(1)).mean()

    # Unit-norm regulariser
    mag_loss = ((torch.norm(pred, dim=1) - 1.0)**2).mean()

    return angle_loss + 0.1 * mag_loss


# ========================= MAIN =========================

def main():
    # --- Load sample IDs ---
    if METADATA_JSON.exists():
        with open(METADATA_JSON) as f:
            meta = json.load(f)
        sample_ids = sorted(
            [int(item["ImgName"]) for item in meta if str(item["ImgName"]).isdigit()]
        )
    else:
        sample_ids = sorted([int(p.stem) for p in SAMPLES_DIR.glob("[0-9]*.bmp")])

    print(f"Found {len(sample_ids)} samples.")
    random.shuffle(sample_ids)
    split     = int(len(sample_ids) * TRAIN_SPLIT)
    train_ids = sample_ids[:split]
    val_ids   = sample_ids[split:]
    print(f"Train: {len(train_ids)}   Val: {len(val_ids)}")

    # --- DataLoaders ---
    train_loader = DataLoader(
        LC_3D_Dataset(train_ids), batch_size=BATCH_SIZE,
        shuffle=True,  num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        LC_3D_Dataset(val_ids), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )

    # --- Model ---
    model     = Director3DNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters : {total_params:,}")
    print(f"Z seed           : {Z_SEED}")
    print(f"Output per sample: (3, {N_Z_SLICES}, {IMG_SIZE}, {IMG_SIZE})")

    # --- Training loop ---
    best_val_loss    = 1e9
    patience_counter = 0

    print("\nStarting Z-depth-aware training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch:3d}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = director_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_loss += director_loss(model(x), y).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        print(
            f"[{epoch:3d}/{EPOCHS}]  "
            f"Train: {train_loss:.3f}°   "
            f"Val: {val_loss:.3f}°   "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✓ Best model saved  (Val = {val_loss:.3f}°)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"\nTraining finished.  Best Val Loss: {best_val_loss:.3f}°")
    print(f"Model saved → {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()