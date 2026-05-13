# -*- coding: utf-8 -*-
"""
Test & Visualisation — 3D LC Director Field Predictor
======================================================
Usage
-----
  # Auto-picks a random sample from Samples/
  python test_director_model.py

  # Test a specific sample ID
  python test_director_model.py --sample 42

  # Specify alternative paths
  python test_director_model.py --sample 5 \
      --samples_dir /path/to/Samples \
      --model /path/to/model.pth

Output
------
  test_results/
    <id>_pom_input.png          — the 4×4 POM composite input image
    <id>_xy_quiver.png          — XY-plane quiver plots (GT vs Pred) at Z=mid
    <id>_xz_quiver.png          — XZ-plane quiver plots at Y=mid
    <id>_yz_quiver.png          — YZ-plane quiver plots at X=mid
    <id>_components_gt.png      — nx / ny / nz component maps (GT)
    <id>_components_pred.png    — nx / ny / nz component maps (Pred)
    <id>_angular_error.png      — per-voxel angular error heatmap (Z slices)
    <id>_z_layers.png           — quiver overlay across all Z slices (GT vs Pred)
    <id>_summary.png            — metrics bar chart + histogram of angular error
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                # headless — works without a display
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import pyvista as pv
from scipy.ndimage import zoom

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  (must match training script)
# ──────────────────────────────────────────────────────────────────────────────
IMG_SIZE   = 80
N_Z_SLICES = 20
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS  — v1 and v2, both kept here so either .pth loads cleanly
# auto_load_model() inspects the state-dict keys and picks the right one.
# ──────────────────────────────────────────────────────────────────────────────

# ── Shared encoder (identical in both versions) ───────────────────────────────

class Encoder2D(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = self._block(16,     base)
        self.enc2 = self._block(base,   base*2)
        self.enc3 = self._block(base*2, base*4)
        self.enc4 = self._block(base*4, base*8)

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


# ── V1 architecture ───────────────────────────────────────────────────────────
#    bridge: 1 × 3D conv, no positional encoding
#    decoder: plain broadcast skips, no z_refine

class ZExpansionBridge_v1(nn.Module):
    def __init__(self, cin, cout, z_seed=5):
        super().__init__()
        self.z_seed = z_seed
        self.project = nn.Sequential(
            nn.Conv2d(cin, cout, 1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv3d(cout, cout, (3,3,3), padding=1),
            nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.project(x)
        x = x.unsqueeze(2).expand(-1, -1, self.z_seed, -1, -1).contiguous()
        return self.refine(x)


class Decoder3D_v1(nn.Module):
    def __init__(self, base=32, z_seed=5, z_target=20):
        super().__init__()
        self.z_target = z_target
        self.up4   = self._upblock(base*8, base*4)
        self.up3   = self._upblock(base*4, base*2)
        self.up2   = self._upblock(base*2, base)
        self.fuse4 = self._fuse3d(base*4 + base*4, base*4)
        self.fuse3 = self._fuse3d(base*2 + base*2, base*2)
        self.fuse2 = self._fuse3d(base   + base,   base)
        self.final = nn.Conv3d(base, 3, 1)

    @staticmethod
    def _upblock(cin, cout):
        return nn.Sequential(
            nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False),
            nn.Conv3d(cin, cout, (1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
        )

    @staticmethod
    def _fuse3d(cin, cout):
        return nn.Sequential(
            nn.Conv3d(cin, cout, (3,3,3), padding=1),
            nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
        )

    @staticmethod
    def _broadcast_skip(skip_2d, z):
        return skip_2d.unsqueeze(2).expand(-1, -1, z, -1, -1).contiguous()

    def forward(self, bottleneck, e1, e2, e3):
        u = self.up4(bottleneck)
        u = self.fuse4(torch.cat([u, self._broadcast_skip(e3, u.shape[2])], dim=1))
        u = self.up3(u)
        u = self.fuse3(torch.cat([u, self._broadcast_skip(e2, u.shape[2])], dim=1))
        u = self.up2(u)
        u = self.fuse2(torch.cat([u, self._broadcast_skip(e1, u.shape[2])], dim=1))
        if u.shape[2] != self.z_target:
            u = torch.nn.functional.interpolate(
                u, size=(self.z_target, u.shape[3], u.shape[4]),
                mode='trilinear', align_corners=False)
        u = self.final(u)
        norm = torch.sqrt(torch.sum(u**2, dim=1, keepdim=True) + 1e-8)
        return u / norm


class Director3DNet_v1(nn.Module):
    def __init__(self, base=32, z_seed=5, z_target=N_Z_SLICES):
        super().__init__()
        self.encoder = Encoder2D(base=base)
        self.bridge  = ZExpansionBridge_v1(cin=base*8, cout=base*8, z_seed=z_seed)
        self.decoder = Decoder3D_v1(base=base, z_seed=z_seed, z_target=z_target)

    def forward(self, x):
        e1, e2, e3, e4 = self.encoder(x)
        return self.decoder(self.bridge(e4), e1, e2, e3)


# ── V2 architecture ───────────────────────────────────────────────────────────
#    bridge: 3 × 3D conv + sinusoidal Z positional encoding
#    decoder: Z-aware skips, z_refine conv after trilinear stretch

def _make_z_encoding(z: int, channels: int, device) -> torch.Tensor:
    """Sinusoidal Z positional encoding, shape (1, channels, z, 1, 1)."""
    pe       = torch.zeros(channels, z)
    position = torch.arange(z, dtype=torch.float).unsqueeze(0)
    div_term = torch.exp(
        torch.arange(0, channels, 2, dtype=torch.float)
        * (-np.log(10000.0) / channels)
    )
    pe[0::2, :] = torch.sin(position * div_term.unsqueeze(1))
    pe[1::2, :] = torch.cos(position * div_term.unsqueeze(1))
    return pe.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)


class ZExpansionBridge_v2(nn.Module):
    def __init__(self, cin, cout, z_seed=10):
        super().__init__()
        self.z_seed = z_seed
        self.cout   = cout
        self.project = nn.Sequential(
            nn.Conv2d(cin, cout, 1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv3d(cout, cout, (3,3,3), padding=1),
            nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, (3,3,3), padding=1),
            nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, (3,3,3), padding=1),
            nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x  = self.project(x)
        x  = x.unsqueeze(2).expand(-1, -1, self.z_seed, -1, -1).contiguous()
        pe = _make_z_encoding(self.z_seed, self.cout, x.device)
        x  = x + pe
        return self.refine(x)


class Decoder3D_v2(nn.Module):
    def __init__(self, base=32, z_seed=10, z_target=20):
        super().__init__()
        self.z_target = z_target
        self.base     = base
        self.up4   = self._upblock(base*8, base*4)
        self.up3   = self._upblock(base*4, base*2)
        self.up2   = self._upblock(base*2, base)
        self.fuse4 = self._fuse3d(base*4 + base*4, base*4)
        self.fuse3 = self._fuse3d(base*2 + base*2, base*2)
        self.fuse2 = self._fuse3d(base   + base,   base)
        self.z_refine = nn.Sequential(
            nn.Conv3d(base, base, (3,3,3), padding=1),
            nn.BatchNorm3d(base), nn.ReLU(inplace=True),
        )
        self.final = nn.Conv3d(base, 3, 1)

    @staticmethod
    def _upblock(cin, cout):
        return nn.Sequential(
            nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False),
            nn.Conv3d(cin, cout, (1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
        )

    @staticmethod
    def _fuse3d(cin, cout):
        return nn.Sequential(
            nn.Conv3d(cin, cout, (3,3,3), padding=1),
            nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
        )

    def _z_aware_skip(self, skip_2d, z, c):
        skip_3d = skip_2d.unsqueeze(2).expand(-1, -1, z, -1, -1).contiguous()
        pe      = _make_z_encoding(z, c, skip_2d.device)
        return skip_3d + pe

    def forward(self, bottleneck, e1, e2, e3):
        u = self.up4(bottleneck)
        u = self.fuse4(torch.cat([u, self._z_aware_skip(e3, u.shape[2], self.base*4)], dim=1))
        u = self.up3(u)
        u = self.fuse3(torch.cat([u, self._z_aware_skip(e2, u.shape[2], self.base*2)], dim=1))
        u = self.up2(u)
        u = self.fuse2(torch.cat([u, self._z_aware_skip(e1, u.shape[2], self.base)],   dim=1))
        if u.shape[2] != self.z_target:
            u = torch.nn.functional.interpolate(
                u, size=(self.z_target, u.shape[3], u.shape[4]),
                mode='trilinear', align_corners=False)
        u = self.z_refine(u)
        u = self.final(u)
        norm = torch.sqrt(torch.sum(u**2, dim=1, keepdim=True) + 1e-8)
        return u / norm


class Director3DNet_v2(nn.Module):
    def __init__(self, base=32, z_seed=10, z_target=N_Z_SLICES):
        super().__init__()
        self.encoder = Encoder2D(base=base)
        self.bridge  = ZExpansionBridge_v2(cin=base*8, cout=base*8, z_seed=z_seed)
        self.decoder = Decoder3D_v2(base=base, z_seed=z_seed, z_target=z_target)

    def forward(self, x):
        e1, e2, e3, e4 = self.encoder(x)
        return self.decoder(self.bridge(e4), e1, e2, e3)


# ── Auto-detection loader ─────────────────────────────────────────────────────

# Keys that exist in v2 but not v1 — presence of any one → v2
_V2_SIGNATURE_KEYS = {"bridge.refine.3.weight", "decoder.z_refine.0.weight"}

def auto_load_model(model_path: Path) -> nn.Module:
    """
    Load a Director3DNet from *model_path* without needing to know which
    version it is.  Sniffs the state-dict keys, instantiates the matching
    architecture, loads the weights, and returns the model in eval mode.
    """
    state = torch.load(model_path, map_location=DEVICE)

    if _V2_SIGNATURE_KEYS & set(state.keys()):
        version = "v2  (Z-depth-aware: 3-layer bridge + z_refine + positional encoding)"
        model   = Director3DNet_v2().to(DEVICE)
    else:
        version = "v1  (original: 1-layer bridge, plain skips)"
        model   = Director3DNet_v1().to(DEVICE)

    model.load_state_dict(state)
    model.eval()
    print(f"  Architecture : {version}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_pom_composite(bmp_path: Path) -> np.ndarray:
    """Returns (16, H, W) float32 [0,1] — each channel is one POM tile."""
    img = np.array(Image.open(bmp_path))
    h, w = img.shape[:2]
    th, tw = h // 4, w // 4
    channels = []
    for row in range(4):
        for col in range(4):
            tile = img[row*th:(row+1)*th, col*tw:(col+1)*tw]
            gray = np.mean(tile, axis=2) if tile.ndim == 3 else tile
            channels.append(gray.astype(np.float32) / 255.0)
    return np.stack(channels, axis=0)


def load_gt_3d(vti_path: Path,
               target_xy: int = IMG_SIZE,
               target_z:  int = N_Z_SLICES) -> np.ndarray:
    """Returns (3, Z, Y, X) float32 unit-normalised director field."""
    grid = pv.read(str(vti_path))
    vec  = grid.point_data['n']
    dims = grid.dimensions                              # (Nx, Ny, Nz)
    n3d  = vec.reshape((dims[2], dims[1], dims[0], 3)).astype(np.float32)
    src_z, src_y, src_x = n3d.shape[:3]
    if (src_z, src_y, src_x) != (target_z, target_xy, target_xy):
        n3d = zoom(n3d, (target_z/src_z, target_xy/src_y, target_xy/src_x, 1), order=1)
    norm = np.sqrt(np.sum(n3d**2, axis=-1, keepdims=True) + 1e-10)
    n3d  = n3d / norm
    return np.transpose(n3d, (3, 0, 1, 2))             # (3, Z, Y, X)


def run_inference(model, pom: np.ndarray) -> np.ndarray:
    """Returns (3, Z, Y, X) float32 predicted director field."""
    model.eval()
    x = torch.from_numpy(pom).unsqueeze(0).float().to(DEVICE)  # (1,16,H,W)
    with torch.no_grad():
        pred = model(x)                                          # (1,3,Z,H,W)
    return pred.squeeze(0).cpu().numpy()                         # (3,Z,Y,X)


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────

def angular_error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Headless angular error (degrees) at every voxel.
    pred, gt : (3, Z, Y, X)   unit vectors
    Returns  : (Z, Y, X)      degrees in [0, 90]
    """
    cos = np.sum(pred * gt, axis=0)                     # (Z, Y, X)
    err = np.degrees(np.arccos(np.abs(cos).clip(0, 1))) # headless: |cos|
    return err


def compute_metrics(err: np.ndarray) -> dict:
    return {
        "Mean angular error (°)":   float(np.mean(err)),
        "Median angular error (°)": float(np.median(err)),
        "Std angular error (°)":    float(np.std(err)),
        "90th percentile (°)":      float(np.percentile(err, 90)),
        "Max angular error (°)":    float(np.max(err)),
        "% voxels < 5°":            float(np.mean(err < 5)  * 100),
        "% voxels < 15°":           float(np.mean(err < 15) * 100),
        "% voxels < 30°":           float(np.mean(err < 30) * 100),
    }


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

# Quiver stride: show every Nth arrow so the plot isn't cluttered
QSTRIDE = 5

def _quiver_ax(ax, U, V, title, cmap_bg=None, bg=None, stride=QSTRIDE):
    """
    Draw a 2D quiver plot of (U, V) director components on ax.

    Coordinate reconciliation
    -------------------------
    - bg (POM image)     : row 0 = top of sample  (image convention)
    - U, V (director)    : row 0 = BOTTOM of cell  (VTK/scientific convention)

    Fix: flip U and V vertically ([::-1]) so row 0 becomes the top,
    matching the image.  Then negate V because imshow with origin='lower'
    has y increasing upward while quiver row-index increases downward.
    The angle colour uses the physically correct (Us, Vs) before negation.
    """
    h, w = U.shape
    X, Y = np.meshgrid(np.arange(0, w, stride), np.arange(0, h, stride))

    # Flip director rows to match image top-down convention
    Us =  U[::-1][::stride, ::stride]
    Vs =  V[::-1][::stride, ::stride]

    # Colour by physical in-plane angle (before axis flip)
    angles = np.degrees(np.arctan2(Vs, Us)) % 180   # headless [0, 180)
    norm   = mcolors.Normalize(vmin=0, vmax=180)

    if bg is not None:
        ax.imshow(bg, cmap='gray', origin='lower', vmin=0, vmax=1, aspect='equal')

    ax.quiver(X, Y, Us, -Vs,        # negate Vs: quiver y is up, row-index is down
              angles,
              norm=norm, cmap=plt.cm.hsv,
              pivot='mid',
              headlength=3, headwidth=3, headaxislength=2.5,
              scale_units='xy', scale=0.15,
              width=0.004)
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xticks([]); ax.set_yticks([])


def fig_pom_input(pom: np.ndarray, sample_id, out_dir: Path):
    """Save the 4×4 POM tile grid as a single figure."""
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    pol_angles = [0, 30, 60, 90]
    ana_angles = [0, 30, 60, 90]
    for r in range(4):
        for c in range(4):
            ax = axes[r, c]
            ax.imshow(pom[r*4+c], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f"P={pol_angles[c]}° A={ana_angles[r]}°", fontsize=7)
            ax.axis('off')
    fig.suptitle(f"Sample {sample_id} — POM Input (16 channels)", fontsize=11, y=1.01)
    plt.tight_layout()
    path = out_dir / f"{sample_id}_pom_input.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig_xy_quiver(pred: np.ndarray, gt: np.ndarray, pom: np.ndarray,
                  sample_id, out_dir: Path):
    """XY quiver at bottom / middle / top Z slices — GT vs Pred side by side."""
    Z = pred.shape[1]
    z_slices = [0, Z//4, Z//2, 3*Z//4, Z-1]
    labels   = ["Z=bottom", f"Z={Z//4}", "Z=mid", f"Z={3*Z//4}", "Z=top"]

    fig, axes = plt.subplots(2, len(z_slices), figsize=(3.5*len(z_slices), 7))
    bg = pom[8]                 # middle POM tile as background guide

    for col, (zi, lbl) in enumerate(zip(z_slices, labels)):
        # GT
        _quiver_ax(axes[0, col],
                   gt[0, zi], gt[1, zi],
                   f"GT — {lbl}",
                   bg=bg)
        # Pred
        _quiver_ax(axes[1, col],
                   pred[0, zi], pred[1, zi],
                   f"Pred — {lbl}",
                   bg=bg)

    axes[0, 0].set_ylabel("Ground Truth", fontsize=10)
    axes[1, 0].set_ylabel("Prediction",   fontsize=10)
    fig.suptitle(f"Sample {sample_id} — XY Director Field (nx, ny arrows)", fontsize=11)
    plt.tight_layout()
    path = out_dir / f"{sample_id}_xy_quiver.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig_xz_quiver(pred: np.ndarray, gt: np.ndarray, sample_id, out_dir: Path):
    """XZ cross-section (nx, nz arrows) at Y = mid."""
    Y_mid = pred.shape[2] // 2
    # Slice: gt/pred shape (3,Z,Y,X) → at Y=Y_mid: (3,Z,X)
    gt_xz   = gt[:,   :, Y_mid, :]    # (3, Z, X)
    pred_xz = pred[:, :, Y_mid, :]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, field, title in zip(axes,
                                 [gt_xz,   pred_xz],
                                 ["Ground Truth XZ (Y=mid)",
                                  "Prediction XZ (Y=mid)"]):
        # nx = field[0], nz = field[2]  (Z axis = vertical, X = horizontal)
        _quiver_ax(ax, field[0], field[2], title)
        ax.set_xlabel("X"); ax.set_ylabel("Z")

    fig.suptitle(f"Sample {sample_id} — XZ Cross-Section (nx, nz arrows)", fontsize=11)
    plt.tight_layout()
    path = out_dir / f"{sample_id}_xz_quiver.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig_yz_quiver(pred: np.ndarray, gt: np.ndarray, sample_id, out_dir: Path):
    """YZ cross-section (ny, nz arrows) at X = mid."""
    X_mid = pred.shape[3] // 2
    gt_yz   = gt[:,   :, :, X_mid]    # (3, Z, Y)
    pred_yz = pred[:, :, :, X_mid]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, field, title in zip(axes,
                                 [gt_yz,   pred_yz],
                                 ["Ground Truth YZ (X=mid)",
                                  "Prediction YZ (X=mid)"]):
        _quiver_ax(ax, field[1], field[2], title)
        ax.set_xlabel("Y"); ax.set_ylabel("Z")

    fig.suptitle(f"Sample {sample_id} — YZ Cross-Section (ny, nz arrows)", fontsize=11)
    plt.tight_layout()
    path = out_dir / f"{sample_id}_yz_quiver.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def _component_fig(field: np.ndarray, title: str, sample_id, out_dir: Path, tag: str):
    """
    3-row × N_col figure showing nx, ny, nz as colour maps at several Z slices.
    field: (3, Z, Y, X)
    """
    Z      = field.shape[1]
    z_idxs = [0, Z//4, Z//2, 3*Z//4, Z-1]
    labels = ["Z=0", f"Z={Z//4}", "Z=mid", f"Z={3*Z//4}", f"Z={Z-1}"]
    cmaps  = ['RdBu_r', 'RdBu_r', 'coolwarm']
    clabels = ['nx', 'ny', 'nz']

    fig, axes = plt.subplots(3, len(z_idxs), figsize=(3*len(z_idxs), 9))
    for row, (comp, cmap, clbl) in enumerate(zip(range(3), cmaps, clabels)):
        for col, (zi, lbl) in enumerate(zip(z_idxs, labels)):
            ax  = axes[row, col]
            im  = ax.imshow(field[comp, zi], cmap=cmap, vmin=-1, vmax=1,
                            origin='lower', aspect='equal')
            ax.set_title(lbl if row == 0 else "", fontsize=8)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(clbl, fontsize=10)
            if col == len(z_idxs) - 1:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, cax=cax)

    fig.suptitle(f"Sample {sample_id} — {title} (nx / ny / nz)", fontsize=11)
    plt.tight_layout()
    path = out_dir / f"{sample_id}_components_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig_angular_error(err: np.ndarray, sample_id, out_dir: Path):
    """Heatmap of angular error for each Z slice."""
    Z   = err.shape[0]
    cols = min(Z, 5)
    rows = (Z + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = np.array(axes).reshape(rows, cols)   # ensure 2-D indexing

    vmax = max(float(np.percentile(err, 95)), 1.0)
    for zi in range(Z):
        r, c = divmod(zi, cols)
        ax   = axes[r, c]
        im   = ax.imshow(err[zi], cmap='hot_r', vmin=0, vmax=vmax,
                         origin='lower', aspect='equal')
        ax.set_title(f"Z={zi}", fontsize=8)
        ax.axis('off')

    # Hide unused subplots
    for zi in range(Z, rows*cols):
        r, c = divmod(zi, cols)
        axes[r, c].axis('off')

    fig.suptitle(
        f"Sample {sample_id} — Headless Angular Error per Z slice\n"
        f"(colour scale 0 – {vmax:.1f}°, clipped at 95th percentile)",
        fontsize=10
    )
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap='hot_r', norm=mcolors.Normalize(0, vmax))
    fig.colorbar(sm, cax=cbar_ax, label='Angular error (°)')

    path = out_dir / f"{sample_id}_angular_error.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig_z_layers(pred: np.ndarray, gt: np.ndarray, sample_id, out_dir: Path):
    """
    Two rows (GT / Pred) × all Z slices showing nx, ny quiver on nz background.
    Useful for a quick full-depth overview.
    """
    Z      = pred.shape[1]
    cols   = min(Z, 10)
    rows_p = 2  # GT row + Pred row

    fig, axes = plt.subplots(rows_p, cols,
                              figsize=(2.5*cols, 5),
                              gridspec_kw={'hspace': 0.05, 'wspace': 0.02})

    for zi in range(cols):
        # Use actual Z index spread evenly if Z > cols
        z_idx = int(zi * Z / cols)

        for row, (field, lbl) in enumerate([(gt, "GT"), (pred, "Pred")]):
            ax = axes[row, zi]
            # nz as colour background
            ax.imshow(field[2, z_idx], cmap='coolwarm',
                      vmin=-1, vmax=1, origin='lower', aspect='equal')
            # nx, ny as quiver overlay
            h, w = field[0, z_idx].shape
            S    = QSTRIDE
            Xg, Yg = np.meshgrid(np.arange(0, w, S), np.arange(0, h, S))
            ax.quiver(Xg, Yg,
                      field[0, z_idx, ::S, ::S],
                      field[1, z_idx, ::S, ::S],
                      pivot='mid', color='k',
                      headlength=3, headwidth=3,
                      scale_units='xy', scale=0.18, width=0.005)
            ax.set_title(f"Z={z_idx}" if row == 0 else "", fontsize=7)
            ax.axis('off')
            if zi == 0:
                ax.set_ylabel(lbl, fontsize=9)

    fig.suptitle(
        f"Sample {sample_id} — All Z Layers  "
        f"(colour=nz, arrows=nx,ny)",
        fontsize=10, y=1.01
    )
    path = out_dir / f"{sample_id}_z_layers.png"
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig_summary(metrics: dict, err: np.ndarray, sample_id, out_dir: Path):
    """Bar chart of key metrics + histogram of per-voxel angular errors."""
    fig = plt.figure(figsize=(14, 5))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.35)

    # --- Left: metric bars ---
    ax_bar = fig.add_subplot(gs[0])
    bar_keys = ["Mean angular error (°)", "Median angular error (°)",
                "Std angular error (°)",  "90th percentile (°)"]
    vals  = [metrics[k] for k in bar_keys]
    short = ["Mean", "Median", "Std", "P90"]
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    bars  = ax_bar.bar(short, vals, color=colors, edgecolor='k', linewidth=0.7)
    for bar, v in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{v:.2f}°", ha='center', va='bottom', fontsize=9)
    ax_bar.set_ylabel("Angular error (°)", fontsize=10)
    ax_bar.set_title(f"Sample {sample_id} — Error Metrics", fontsize=10)
    ax_bar.set_ylim(0, max(vals) * 1.25 + 1)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.5)

    # Accuracy thresholds text box
    txt = (
        f"  < 5°  : {metrics['% voxels < 5°']:.1f}% of voxels\n"
        f"  < 15° : {metrics['% voxels < 15°']:.1f}% of voxels\n"
        f"  < 30° : {metrics['% voxels < 30°']:.1f}% of voxels"
    )
    ax_bar.text(0.97, 0.97, txt,
                transform=ax_bar.transAxes,
                va='top', ha='right', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    # --- Right: histogram ---
    ax_hist = fig.add_subplot(gs[1])
    flat_err = err.ravel()
    ax_hist.hist(flat_err, bins=90, range=(0, 90),
                 color='steelblue', edgecolor='none', alpha=0.85, density=True)
    ax_hist.axvline(metrics["Mean angular error (°)"],   color='red',    lw=1.5,
                    linestyle='--', label=f"Mean {metrics['Mean angular error (°)']:.1f}°")
    ax_hist.axvline(metrics["Median angular error (°)"], color='orange', lw=1.5,
                    linestyle='-',  label=f"Median {metrics['Median angular error (°)']:.1f}°")
    ax_hist.set_xlabel("Headless angular error (°)", fontsize=10)
    ax_hist.set_ylabel("Density", fontsize=10)
    ax_hist.set_title(f"Sample {sample_id} — Error Distribution", fontsize=10)
    ax_hist.legend(fontsize=9)
    ax_hist.grid(linestyle='--', alpha=0.4)

    path = out_dir / f"{sample_id}_summary.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test & visualise 3D LC director predictor on a single sample."
    )
    parser.add_argument("--sample",      type=int,   default=None,
                        help="Sample ID to test (default: random)")
    parser.add_argument("--samples_dir", type=str,   default="Samples",
                        help="Path to Samples directory (default: Samples/)")
    parser.add_argument("--model",       type=str,   default=None,
                        help="Path to .pth model weights "
                             "(default: <samples_dir>/director_model_full3d.pth)")
    parser.add_argument("--out_dir",     type=str,   default="test_results",
                        help="Output directory for plots (default: test_results/)")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # ── Resolve model path ───────────────────────────────────────────────────
    if args.model:
        # Explicit path supplied by the user
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"ERROR: Model weights not found at {model_path}")
            sys.exit(1)
    else:
        # Auto-detect: prefer the new name, fall back to old name, then scan
        candidates = [
            samples_dir / "director_model_full3d_ver2.pth",       # new training script
        ]
        # Also pick up any other .pth files sitting in the samples dir
        candidates += sorted(samples_dir.glob("*.pth"))

        # Deduplicate while preserving order
        seen = set()
        candidates = [p for p in candidates if not (p in seen or seen.add(p))]

        model_path = next((p for p in candidates if p.exists()), None)

        if model_path is None:
            pth_files = list(samples_dir.glob("*.pth"))
            print(f"ERROR: No model weights found in {samples_dir}/")
            if pth_files:
                print(f"  .pth files present: {[p.name for p in pth_files]}")
                print(f"  Pass the correct path with:  --model Samples/<filename>.pth")
            else:
                print("  No .pth files found at all. Has training finished?")
            sys.exit(1)

        print(f"Auto-detected model: {model_path}")

    # ── Pick sample ──────────────────────────────────────────────────────────
    available = sorted([int(p.stem) for p in samples_dir.glob("[0-9]*.bmp")])
    if not available:
        print(f"ERROR: No .bmp files found in {samples_dir}")
        sys.exit(1)

    if args.sample is not None:
        sid = args.sample
        if sid not in available:
            print(f"ERROR: Sample {sid} not found. Available: {available[:10]} …")
            sys.exit(1)
    else:
        sid = random.choice(available)
        print(f"Randomly selected sample: {sid}")

    bmp_path = samples_dir / f"{sid}.bmp"
    vti_path = samples_dir / f"{sid}.vti"

    if not bmp_path.exists():
        print(f"ERROR: {bmp_path} not found."); sys.exit(1)
    if not vti_path.exists():
        print(f"ERROR: {vti_path} not found."); sys.exit(1)

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"Loading model from {model_path} …")
    model = auto_load_model(model_path)
    print(f"Model loaded.  Device: {DEVICE}")

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading sample {sid} …")
    pom  = load_pom_composite(bmp_path)       # (16, H, W)
    gt   = load_gt_3d(vti_path)               # (3, Z, Y, X)
    pred = run_inference(model, pom)           # (3, Z, Y, X)

    print(f"  POM shape   : {pom.shape}")
    print(f"  GT shape    : {gt.shape}")
    print(f"  Pred shape  : {pred.shape}")

    # ── Metrics ──────────────────────────────────────────────────────────────
    err     = angular_error_map(pred, gt)      # (Z, Y, X)
    metrics = compute_metrics(err)

    print("\n──────────────── Metrics ────────────────")
    for k, v in metrics.items():
        print(f"  {k:<30s}: {v:.3f}")
    print("─────────────────────────────────────────\n")

    # ── Plots ────────────────────────────────────────────────────────────────
    print(f"Saving plots to {out_dir}/ …")

    fig_pom_input(pom, sid, out_dir)
    fig_xy_quiver(pred, gt, pom, sid, out_dir)
    fig_xz_quiver(pred, gt, sid, out_dir)
    fig_yz_quiver(pred, gt, sid, out_dir)
    _component_fig(gt,   "Ground Truth", sid, out_dir, "gt")
    _component_fig(pred, "Prediction",   sid, out_dir, "pred")
    fig_angular_error(err, sid, out_dir)
    fig_z_layers(pred, gt, sid, out_dir)
    fig_summary(metrics, err, sid, out_dir)

    print(f"\nDone! All plots saved to: {out_dir}/")


if __name__ == "__main__":
    main()