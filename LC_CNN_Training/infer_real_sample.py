# -*- coding: utf-8 -*-
"""
Created on Thu May  7 23:53:58 2026

@author: tomla
"""

"""
Real-Sample Inference — 3D LC Director Field Predictor
=======================================================
Run the trained model on real experimental POM images.
No VTI ground truth required; prediction only.

Supported input modes
---------------------
  MODE A — Single image (any format: TIFF, PNG, BMP, JPG, …)
    The image is treated as ONE POM acquisition.  Because the model
    expects 16 channels (one per polariser/analyser angle pair), the
    single image is replicated across all 16 channels.  This gives a
    valid inference but naturally loses angle-dependent information.

    python infer_real_sample.py --image my_pom.tif

  MODE B — Folder of images (2–16 images at different angle settings)
    Images are sorted by filename and assigned to channels in order.
    If fewer than 16 are provided the remaining channels are filled by
    cycling through the available images (nearest-angle approximation).

    python infer_real_sample.py --image_dir pom_angles/

  MODE C — Pre-tiled composite (same 4×4 format as training data)
    If your image is already the 4×4 composite (H×4, W×4 pixels), pass
    --composite to skip the tiling step.

    python infer_real_sample.py --image composite.tif --composite

Resolution handling
-------------------
Each tile must be 80×80 px for the model.  If the input (or each tile
after splitting) is a different size, the script offers three strategies
selected with --resize:

  crop   — centre-crop to the largest 80×80 square (default)
  resize — bicubic resize to 80×80  (distorts aspect ratio)
  tile   — divide the image into non-overlapping 80×80 patches and run
            inference on each patch independently; results are stitched
            back into a full-resolution prediction map

Usage examples
--------------
  python infer_real_sample.py --image experiment.tif
  python infer_real_sample.py --image experiment.tif --resize resize
  python infer_real_sample.py --image experiment.tif --resize tile
  python infer_real_sample.py --image_dir angles/ --resize crop
  python infer_real_sample.py --image composite.tif --composite
  python infer_real_sample.py --image experiment.tif \\
      --model Samples/director_model_full3d_ver2.pth \\
      --out_dir results/

Output  (saved to --out_dir, default: inference_results/)
-------
  <stem>_input_channels.png    — the 16 channels fed to the model
  <stem>_xy_quiver.png         — XY director field at 5 Z depths
  <stem>_xz_quiver.png         — XZ cross-section at Y=mid
  <stem>_yz_quiver.png         — YZ cross-section at X=mid
  <stem>_components.png        — nx / ny / nz colour maps per Z slice
  <stem>_z_layers.png          — all Z layers overview (nz=colour, nx/ny=arrows)
  <stem>_director_field.npz    — raw prediction array, shape (3, Z, Y, X)
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import zoom

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
TILE_SIZE  = 80        # spatial resolution expected by the model
N_CHANNELS = 16        # polariser/analyser channel count
N_Z_SLICES = 20        # Z depth of the predicted volume
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SUPPORTED_EXTS = {".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg"}

# ──────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS  (v1 + v2 with auto-detection — keep in sync with training)
# ──────────────────────────────────────────────────────────────────────────────

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
            u = F.interpolate(u, size=(self.z_target, u.shape[3], u.shape[4]),
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


def _make_z_encoding(z, channels, device):
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
            nn.Conv3d(cout, cout, (3,3,3), padding=1), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, (3,3,3), padding=1), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, (3,3,3), padding=1), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
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
            u = F.interpolate(u, size=(self.z_target, u.shape[3], u.shape[4]),
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


_V2_KEYS = {"bridge.refine.3.weight", "decoder.z_refine.0.weight"}

def auto_load_model(model_path: Path) -> nn.Module:
    state = torch.load(model_path, map_location=DEVICE)
    if _V2_KEYS & set(state.keys()):
        version = "v2  (Z-depth-aware)"
        model   = Director3DNet_v2().to(DEVICE)
    else:
        version = "v1  (original)"
        model   = Director3DNet_v1().to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"  Architecture : {version}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# IMAGE LOADING  — handles any format, any channel count
# ──────────────────────────────────────────────────────────────────────────────

def _open_as_gray(path: Path) -> np.ndarray:
    """
    Open any supported image file and return a 2D float32 array in [0, 1].

    Handles:
    - 8-bit  grayscale / RGB / RGBA
    - 16-bit grayscale TIFF  (common in microscopy)
    - 32-bit float TIFF
    Multi-page TIFFs are flattened to their first page.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress TIFF metadata warnings
        img = Image.open(path)

        # Multi-frame TIFF: take first frame
        try:
            img.seek(0)
        except AttributeError:
            pass

        arr = np.array(img)

    # Collapse to single channel
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)   # RGB / RGBA → luminance

    arr = arr.astype(np.float32)

    # Normalise to [0, 1] based on dtype
    if arr.dtype == np.float32 and arr.max() <= 1.0:
        pass                              # already in [0,1]
    elif arr.max() > 1.0:
        arr = arr / arr.max()             # handles 8-bit (255), 16-bit (65535), etc.

    return arr


def load_images_from_folder(folder: Path) -> list[np.ndarray]:
    """
    Load all supported image files from a folder, sorted by filename.
    Returns a list of 2D float32 arrays.
    """
    paths = sorted([p for p in folder.iterdir()
                    if p.suffix.lower() in SUPPORTED_EXTS])
    if not paths:
        print(f"ERROR: No supported images found in {folder}")
        sys.exit(1)
    print(f"  Found {len(paths)} image(s) in {folder}:")
    for p in paths:
        print(f"    {p.name}")
    return [_open_as_gray(p) for p in paths]


# ──────────────────────────────────────────────────────────────────────────────
# CHANNEL ASSEMBLY  — map raw images → (16, 80, 80) model input
# ──────────────────────────────────────────────────────────────────────────────

def _centre_crop(arr: np.ndarray, size: int) -> np.ndarray:
    h, w   = arr.shape
    mh, mw = min(h, size), min(w, size)
    y0 = (h - mh) // 2
    x0 = (w - mw) // 2
    crop = arr[y0:y0+mh, x0:x0+mw]
    if crop.shape != (size, size):
        # pad if smaller than size
        out    = np.zeros((size, size), dtype=np.float32)
        py, px = (size - mh) // 2, (size - mw) // 2
        out[py:py+mh, px:px+mw] = crop
        return out
    return crop


def _bicubic_resize(arr: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray((arr * 65535).astype(np.uint16))
    img = img.resize((size, size), Image.BICUBIC)
    return np.array(img).astype(np.float32) / 65535.0


def _to_tile(arr: np.ndarray, resize_mode: str) -> np.ndarray:
    """Reduce a single 2D image to (TILE_SIZE, TILE_SIZE) using resize_mode."""
    if arr.shape == (TILE_SIZE, TILE_SIZE):
        return arr
    if resize_mode == "crop":
        return _centre_crop(arr, TILE_SIZE)
    elif resize_mode == "resize":
        return _bicubic_resize(arr, TILE_SIZE)
    else:
        raise ValueError(f"Unknown resize mode '{resize_mode}' in _to_tile")


def assemble_channels(images: list[np.ndarray],
                      resize_mode: str) -> np.ndarray:
    """
    Map 1–16 raw images → float32 array of shape (16, TILE_SIZE, TILE_SIZE).

    If fewer than 16 images are given they are cycled (nearest-angle fill).
    If more than 16, only the first 16 are used with a warning.
    """
    n = len(images)
    if n > N_CHANNELS:
        warnings.warn(f"Got {n} images but model expects {N_CHANNELS}. "
                      f"Using first {N_CHANNELS}.")
        images = images[:N_CHANNELS]
        n      = N_CHANNELS

    # Cycle to fill 16 channels
    tiles = []
    for i in range(N_CHANNELS):
        raw  = images[i % n]
        tile = _to_tile(raw, resize_mode)
        tiles.append(tile)

    channels = np.stack(tiles, axis=0)   # (16, H, W)
    print(f"  Channel assembly: {n} image(s) → {N_CHANNELS} channels  "
          f"(each {TILE_SIZE}×{TILE_SIZE} px, mode='{resize_mode}')")
    return channels


def split_composite(img: np.ndarray) -> np.ndarray:
    """
    Split a pre-tiled 4×4 composite image into 16 channels.
    Expects img shape (4*H, 4*W) or (4*H, 4*W, 3).
    """
    if img.ndim == 3:
        img = img.mean(axis=2)
    h, w = img.shape
    th, tw = h // 4, w // 4
    channels = []
    for row in range(4):
        for col in range(4):
            tile = img[row*th:(row+1)*th, col*tw:(col+1)*tw]
            # Resize each tile to TILE_SIZE if needed
            if tile.shape != (TILE_SIZE, TILE_SIZE):
                tile = _bicubic_resize(tile, TILE_SIZE)
            channels.append(tile.astype(np.float32))
    return np.stack(channels, axis=0)   # (16, TILE_SIZE, TILE_SIZE)


# ──────────────────────────────────────────────────────────────────────────────
# TILED INFERENCE  — for large images where resize would lose detail
# ──────────────────────────────────────────────────────────────────────────────

def infer_tiled(model: nn.Module,
                images: list[np.ndarray]) -> tuple[np.ndarray, tuple]:
    """
    Divide each image into non-overlapping TILE_SIZE patches and run
    inference on every patch independently.  Results are stitched back
    into a full-resolution prediction volume.

    Returns
    -------
    pred_full  : (3, Z, rows*TILE_SIZE, cols*TILE_SIZE)
    grid_shape : (rows, cols)
    """
    # All images assumed same resolution — use first
    H, W    = images[0].shape
    rows    = H // TILE_SIZE
    cols    = W // TILE_SIZE

    if rows == 0 or cols == 0:
        print(f"  WARNING: image ({H}×{W}) smaller than tile size ({TILE_SIZE}). "
              f"Falling back to 'resize' mode.")
        channels = assemble_channels(images, "resize")
        pred     = _run_model(model, channels)
        return pred, (1, 1)

    print(f"  Tiled inference: {H}×{W} → {rows}×{cols} grid of "
          f"{TILE_SIZE}×{TILE_SIZE} patches ({rows*cols} total)")

    pred_grid = np.zeros(
        (3, N_Z_SLICES, rows * TILE_SIZE, cols * TILE_SIZE),
        dtype=np.float32
    )

    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * TILE_SIZE, c * TILE_SIZE
            patch_imgs = [img[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE] for img in images]
            channels   = assemble_channels(patch_imgs, "crop")
            pred_patch = _run_model(model, channels)   # (3, Z, 80, 80)
            pred_grid[:, :,
                      y0:y0+TILE_SIZE,
                      x0:x0+TILE_SIZE] = pred_patch
            print(f"    patch [{r},{c}] done", end="\r")

    print()
    return pred_grid, (rows, cols)


def _run_model(model: nn.Module, channels: np.ndarray) -> np.ndarray:
    """Run model on a (16, H, W) float32 array; return (3, Z, H, W)."""
    x = torch.from_numpy(channels).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        pred = model(x)
    return pred.squeeze(0).cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# VISUALISATION  (prediction only, no ground truth)
# ──────────────────────────────────────────────────────────────────────────────

QSTRIDE = 5   # quiver arrow stride

def _quiver_ax(ax, U, V, title, bg=None):
    h, w = U.shape
    S    = QSTRIDE
    X, Y = np.meshgrid(np.arange(0, w, S), np.arange(0, h, S))
    Us, Vs = U[::S, ::S], V[::S, ::S]
    angles = np.degrees(np.arctan2(Vs, Us)) % 180
    norm   = mcolors.Normalize(vmin=0, vmax=180)
    if bg is not None:
        ax.imshow(bg, cmap='gray', origin='lower', vmin=0, vmax=1, aspect='equal')
    ax.quiver(X, Y, Us, Vs, angles, norm=norm, cmap=plt.cm.hsv,
              pivot='mid', headlength=3, headwidth=3, headaxislength=2.5,
              scale_units='xy', scale=0.15, width=0.004)
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xticks([]); ax.set_yticks([])


def fig_input_channels(channels: np.ndarray, stem: str, out_dir: Path):
    """Save the 16 input channels as a 4×4 grid."""
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    pol = [0, 30, 60, 90]
    ana = [0, 30, 60, 90]
    for r in range(4):
        for c in range(4):
            ax = axes[r, c]
            ax.imshow(channels[r*4+c], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f"P={pol[c]}° A={ana[r]}°", fontsize=7)
            ax.axis('off')
    fig.suptitle("Model Input — 16 channels", fontsize=11, y=1.01)
    plt.tight_layout()
    path = out_dir / f"{stem}_input_channels.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig_xy_quiver(pred: np.ndarray, channels: np.ndarray, stem: str, out_dir: Path):
    """XY quiver at 5 Z depths."""
    Z        = pred.shape[1]
    z_idxs   = [0, Z//4, Z//2, 3*Z//4, Z-1]
    z_labels = ["Z=bottom", f"Z={Z//4}", "Z=mid", f"Z={3*Z//4}", "Z=top"]
    bg       = channels[8]

    fig, axes = plt.subplots(1, len(z_idxs), figsize=(3.5*len(z_idxs), 4))
    for ax, zi, lbl in zip(axes, z_idxs, z_labels):
        _quiver_ax(ax, pred[0, zi], pred[1, zi], lbl, bg=bg)

    fig.suptitle("Predicted Director — XY Plane (nx, ny arrows)", fontsize=11)
    plt.tight_layout()
    path = out_dir / f"{stem}_xy_quiver.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig_xz_quiver(pred: np.ndarray, stem: str, out_dir: Path):
    """XZ cross-section (nx, nz) at Y=mid."""
    Y_mid = pred.shape[2] // 2
    xz    = pred[:, :, Y_mid, :]      # (3, Z, X)
    fig, ax = plt.subplots(figsize=(8, 4))
    _quiver_ax(ax, xz[0], xz[2], "XZ Cross-Section — nx, nz  (Y=mid)")
    ax.set_xlabel("X"); ax.set_ylabel("Z")
    fig.suptitle("Predicted Director — XZ Plane", fontsize=11)
    plt.tight_layout()
    path = out_dir / f"{stem}_xz_quiver.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig_yz_quiver(pred: np.ndarray, stem: str, out_dir: Path):
    """YZ cross-section (ny, nz) at X=mid."""
    X_mid = pred.shape[3] // 2
    yz    = pred[:, :, :, X_mid]      # (3, Z, Y)
    fig, ax = plt.subplots(figsize=(8, 4))
    _quiver_ax(ax, yz[1], yz[2], "YZ Cross-Section — ny, nz  (X=mid)")
    ax.set_xlabel("Y"); ax.set_ylabel("Z")
    fig.suptitle("Predicted Director — YZ Plane", fontsize=11)
    plt.tight_layout()
    path = out_dir / f"{stem}_yz_quiver.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig_components(pred: np.ndarray, stem: str, out_dir: Path):
    """Colour maps of nx, ny, nz across Z slices."""
    Z       = pred.shape[1]
    z_idxs  = [0, Z//4, Z//2, 3*Z//4, Z-1]
    z_lbls  = ["Z=0", f"Z={Z//4}", "Z=mid", f"Z={3*Z//4}", f"Z={Z-1}"]
    clabels = ["nx", "ny", "nz"]

    fig, axes = plt.subplots(3, len(z_idxs), figsize=(3*len(z_idxs), 9))
    for row, clbl in enumerate(clabels):
        for col, (zi, zlbl) in enumerate(zip(z_idxs, z_lbls)):
            ax = axes[row, col]
            im = ax.imshow(pred[row, zi], cmap='RdBu_r', vmin=-1, vmax=1,
                           origin='lower', aspect='equal')
            ax.set_title(zlbl if row == 0 else "", fontsize=8)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(clbl, fontsize=10)
            if col == len(z_idxs) - 1:
                divider = make_axes_locatable(ax)
                cax     = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, cax=cax)

    fig.suptitle("Predicted Director Components (nx / ny / nz)", fontsize=11)
    plt.tight_layout()
    path = out_dir / f"{stem}_components.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig_z_layers(pred: np.ndarray, stem: str, out_dir: Path):
    """All Z layers: nz as colour, nx/ny as arrows."""
    Z    = pred.shape[1]
    cols = min(Z, 10)

    fig, axes = plt.subplots(1, cols, figsize=(2.5*cols, 3),
                              gridspec_kw={'wspace': 0.02})

    for ci in range(cols):
        zi  = int(ci * Z / cols)
        ax  = axes[ci]
        ax.imshow(pred[2, zi], cmap='coolwarm', vmin=-1, vmax=1,
                  origin='lower', aspect='equal')
        h, w = pred[0, zi].shape
        S    = QSTRIDE
        Xg, Yg = np.meshgrid(np.arange(0, w, S), np.arange(0, h, S))
        ax.quiver(Xg, Yg, pred[0, zi, ::S, ::S], pred[1, zi, ::S, ::S],
                  pivot='mid', color='k', headlength=3, headwidth=3,
                  scale_units='xy', scale=0.18, width=0.005)
        ax.set_title(f"Z={zi}", fontsize=7)
        ax.axis('off')

    fig.suptitle("Predicted Director — All Z Layers  (colour=nz, arrows=nx,ny)",
                 fontsize=10, y=1.02)
    path = out_dir / f"{stem}_z_layers.png"
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig_tiled_overview(pred: np.ndarray, grid: tuple,
                       stem: str, out_dir: Path):
    """
    For tiled inference: full-FOV nz heatmap at mid-Z with arrows,
    showing the full stitched spatial extent.
    """
    rows, cols = grid
    if rows == 1 and cols == 1:
        return   # single tile — already shown in z_layers

    Z    = pred.shape[1]
    zi   = Z // 2
    S    = QSTRIDE * 2   # coarser stride for the wide FOV

    fig, ax = plt.subplots(figsize=(min(cols * 2, 18), min(rows * 2, 14)))
    ax.imshow(pred[2, zi], cmap='coolwarm', vmin=-1, vmax=1,
              origin='lower', aspect='equal')
    H_full, W_full = pred.shape[2], pred.shape[3]
    Xg, Yg = np.meshgrid(np.arange(0, W_full, S), np.arange(0, H_full, S))
    ax.quiver(Xg, Yg,
              pred[0, zi, ::S, ::S],
              pred[1, zi, ::S, ::S],
              pivot='mid', color='k', headlength=3, headwidth=3,
              scale_units='xy', scale=0.20, width=0.002)
    ax.set_title(f"Full FOV — Z=mid  ({rows}×{cols} tile grid)", fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    path = out_dir / f"{stem}_full_fov.png"
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run director-field inference on real experimental POM images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",     type=str,
                     help="Path to a single POM image (any format)")
    src.add_argument("--image_dir", type=str,
                     help="Folder containing 2–16 POM images at different angles")

    parser.add_argument("--composite", action="store_true",
                        help="Input is already a 4×4 composite (same as training data)")
    parser.add_argument("--resize", choices=["crop", "resize", "tile"],
                        default="crop",
                        help="How to handle non-80×80 images: "
                             "crop (centre-crop), resize (bicubic), "
                             "tile (patch grid, largest FOV)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to .pth weights (auto-detected if not given)")
    parser.add_argument("--samples_dir", type=str, default="Samples",
                        help="Where to look for weights if --model not given")
    parser.add_argument("--out_dir", type=str, default="inference_results",
                        help="Output directory (default: inference_results/)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve model weights ─────────────────────────────────────────────────
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"ERROR: model not found at {model_path}"); sys.exit(1)
    else:
        samples_dir = Path(args.samples_dir)
        candidates  = [
            samples_dir / "director_model_full3d.pth",
            samples_dir / "director_model_3d_improved.pth",
        ] + sorted(samples_dir.glob("*.pth"))
        seen  = set()
        candidates = [p for p in candidates if not (p in seen or seen.add(p))]
        model_path = next((p for p in candidates if p.exists()), None)
        if model_path is None:
            print(f"ERROR: No .pth found in {samples_dir}. "
                  f"Use --model to specify the path."); sys.exit(1)
        print(f"Auto-detected model: {model_path}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model …")
    model = auto_load_model(model_path)
    print(f"Device: {DEVICE}")

    # ── Load input images ─────────────────────────────────────────────────────
    if args.image_dir:
        src_path = Path(args.image_dir)
        stem     = src_path.stem
        images   = load_images_from_folder(src_path)
        print(f"  Input: {len(images)} image(s) from folder '{src_path}'")
    else:
        src_path = Path(args.image)
        stem     = src_path.stem
        if not src_path.exists():
            print(f"ERROR: {src_path} not found."); sys.exit(1)
        raw = _open_as_gray(src_path)
        print(f"  Input: '{src_path.name}'  shape={raw.shape}  "
              f"dtype=float32  range=[{raw.min():.3f}, {raw.max():.3f}]")
        images = [raw]

    # ── Assemble model input ──────────────────────────────────────────────────
    if args.composite:
        # Image is already a 4×4 tile grid
        print("  Mode: composite split")
        raw_img  = images[0] if len(images) == 1 else np.mean(
            np.stack(images), axis=0)
        channels = split_composite(raw_img)
        pred     = _run_model(model, channels)
        grid     = (1, 1)

    elif args.resize == "tile":
        # Tiled inference: preserves full FOV for large images
        print("  Mode: tiled inference")
        pred, grid = infer_tiled(model, images)
        # For per-tile channel viz, use first tile
        tile_imgs  = [img[:TILE_SIZE, :TILE_SIZE] for img in images]
        channels   = assemble_channels(tile_imgs, "crop")

    else:
        # Single inference after crop/resize
        print(f"  Mode: single inference  (resize='{args.resize}')")
        channels = assemble_channels(images, args.resize)
        pred     = _run_model(model, channels)
        grid     = (1, 1)

    print(f"  Prediction shape: {pred.shape}  "
          f"(3 components, {pred.shape[1]} Z, "
          f"{pred.shape[2]}×{pred.shape[3]} XY)")

    # ── Save raw prediction ───────────────────────────────────────────────────
    npz_path = out_dir / f"{stem}_director_field.npz"
    np.savez_compressed(npz_path, director=pred,
                        description="shape=(3,Z,Y,X) — (nx,ny,nz) unit vectors")
    print(f"  Saved: {npz_path.name}  (load with np.load(path)['director'])")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\nSaving plots to {out_dir}/ …")
    fig_input_channels(channels, stem, out_dir)
    fig_xy_quiver(pred, channels, stem, out_dir)
    fig_xz_quiver(pred, stem, out_dir)
    fig_yz_quiver(pred, stem, out_dir)
    fig_components(pred, stem, out_dir)
    fig_z_layers(pred, stem, out_dir)
    if args.resize == "tile":
        fig_tiled_overview(pred, grid, stem, out_dir)

    print(f"\nDone!  All outputs saved to: {out_dir}/")
    print(f"  Load prediction: np.load('{npz_path}')['director']  → shape (3,Z,Y,X)")


if __name__ == "__main__":
    main()