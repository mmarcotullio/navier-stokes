# inverse_tool.py
# Self-contained inverse modeling module.
# Architecture copied from inverse/model.py; preprocessing from inverse/dataset.py.

import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Module-level paths and constants ─────────────────────────────────────────
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_VIS_DIR     = os.path.dirname(_THIS_DIR)
_MODEL_PATH  = os.path.join(_VIS_DIR, "models", "cnn_best_inverse.pt")
_STATS_PATH  = os.path.join(_VIS_DIR, "models", "field_stats_dimless_voxel.json")
_RE_LOG_MIN  = math.log(100.0)
_RE_LOG_MAX  = math.log(1000.0)
_inverse_model = None
_field_stats   = None


# ── Architecture ──────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention (Hu et al. 2018).
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        w = self.pool(x).view(B, C)
        w = self.fc(w).view(B, C, 1, 1, 1)
        return x * w


class ResBlock3d(nn.Module):
    """
    Two-conv residual block with anisotropic kernels and SE channel attention.
    """

    def __init__(
        self,
        in_ch:  int,
        out_ch: int,
        k_ax:   int = 5,
        pool:   "nn.Module | None" = None,
    ):
        super().__init__()
        pad_ax = k_ax // 2
        self.conv1 = nn.Conv3d(
            in_ch, out_ch, (k_ax, 3, 3),
            padding=(pad_ax, 1, 1), bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(
            out_ch, out_ch, (k_ax, 3, 3),
            padding=(pad_ax, 1, 1), bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.skip = (
            nn.Conv3d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )
        self.se   = SEBlock(out_ch)
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        h = F.relu(h + self.skip(x), inplace=True)
        h = self.se(h)
        if self.pool is not None:
            h = self.pool(h)
        return h


class CNN3DInverse(nn.Module):
    """
    3D residual CNN for inverse CFD parameter prediction.

    Input:  (B, in_channels, Nx, Ny, Nz)
    Output: (B, 1) — Re_log_norm prediction
    """

    def __init__(self, in_channels: int = 6):
        super().__init__()
        self.blocks = nn.Sequential(
            ResBlock3d(in_channels,  32, k_ax=3, pool=None),
            ResBlock3d(32,   64, k_ax=5, pool=nn.MaxPool3d((2, 2, 2))),
            ResBlock3d(64,  128, k_ax=5, pool=nn.MaxPool3d((2, 2, 2))),
            ResBlock3d(128, 256, k_ax=5, pool=nn.MaxPool3d((2, 1, 1))),
            ResBlock3d(256, 256, k_ax=5, pool=nn.MaxPool3d((2, 1, 1))),
        )
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.gap(x).flatten(1)
        return self.head(x)


# ── Finite-difference helper ──────────────────────────────────────────────────

def _central_diff(arr: np.ndarray, ax: int, h: float) -> np.ndarray:
    """
    Second-order finite differences along axis ax with uniform spacing h.
    Interior: central differences. Boundaries: one-sided 2nd-order.
    arr shape: (Nx, Ny, Nz).  ax ∈ {0, 1, 2}.
    """
    out = np.empty_like(arr)
    n = arr.shape[ax]

    def _sl(axis, idx_or_slice):
        s = [slice(None)] * 3
        s[axis] = idx_or_slice
        return tuple(s)

    if n >= 3:
        # Interior: central difference
        out[_sl(ax, slice(1, -1))] = (
            arr[_sl(ax, slice(2, None))] - arr[_sl(ax, slice(None, -2))]
        ) / (2.0 * h)
        # Left boundary: 2nd-order forward
        out[_sl(ax, 0)] = (
            -3.0 * arr[_sl(ax, 0)]
            + 4.0 * arr[_sl(ax, 1)]
            -       arr[_sl(ax, 2)]
        ) / (2.0 * h)
        # Right boundary: 2nd-order backward
        out[_sl(ax, -1)] = (
             3.0 * arr[_sl(ax, -1)]
            - 4.0 * arr[_sl(ax, -2)]
            +       arr[_sl(ax, -3)]
        ) / (2.0 * h)
    elif n == 2:
        out[_sl(ax, 0)]  = (arr[_sl(ax, 1)] - arr[_sl(ax, 0)]) / h
        out[_sl(ax, -1)] = out[_sl(ax, 0)]
    else:
        out[:] = 0.0

    return out


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(device: str):
    """Lazy singleton: load inverse model and field stats once."""
    global _inverse_model, _field_stats
    if _inverse_model is None:
        with open(_STATS_PATH) as f:
            _field_stats = json.load(f)
        model = CNN3DInverse(in_channels=6)
        model.load_state_dict(torch.load(_MODEL_PATH, map_location=device, weights_only=True))
        model.to(device).eval()
        _inverse_model = model
    return _inverse_model, _field_stats


# ── D4 Test-Time Augmentation ─────────────────────────────────────────────────
# The pipe cross-section has approximate D4 (square) symmetry in the y-z plane.
# Averaging predictions over 2 variants (identity + y-flip) reduces variance.

def _d4_augment(vt: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """Apply one element of the D4 group (y-z plane) to a (1,6,Nx,Ny,Nz) tensor.

    Channels: 0=ux, 1=uy, 2=uz, 3=p_c, 4=dp_dx, 5=vort_mag
    Spatial dims: 2=x (axial), 3=y, 4=z
    """
    vt = vt.clone()
    if flip:
        vt       = torch.flip(vt, dims=[3])   # y → -y
        vt[:, 1] = -vt[:, 1]                  # uy → -uy
    if k > 0:
        vt = torch.rot90(vt, k=k, dims=[3, 4])  # rotate spatially in y-z
        for _ in range(k):                       # rotate velocity vector to match
            uy       = vt[:, 1].clone()
            vt[:, 1] = -vt[:, 2]
            vt[:, 2] =  uy
    return vt


def _tta_predict(model: torch.nn.Module, vt: torch.Tensor, device: str) -> float:
    """Average CNN predictions over 2 augmentations: identity + y-flip (k=0)."""
    preds = []
    with torch.no_grad():
        for flip in (False, True):
            aug = _d4_augment(vt, 0, flip).to(device)
            preds.append(float(model(aug).cpu().item()))
    return float(np.mean(preds))


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_from_voxel_raw(voxel_raw: np.ndarray, U_in: float, device: str) -> float:
    """
    Predict Re from a raw 6-channel physical-unit voxel (fno_to_voxel output).
    Replicates collect_re_predictions.py post-FNO processing exactly.

    voxel_raw: (6, Nx, Ny, Nz) float32 — channels: ux,uy,uz,p_c,dp_dx,vort_mag
    Non-fluid voxels must already be zeroed (as fno_to_voxel does).
    No pressure re-centering: FNO output is already mean-centred.
    """
    model, stats = load_model(device)
    means = np.array(stats["mean"], dtype=np.float32)
    stds  = np.array(stats["std"],  dtype=np.float32)

    voxel = voxel_raw.copy()
    voxel[0] /= U_in        # ux / U_in
    voxel[1] /= U_in        # uy / U_in
    voxel[2] /= U_in        # uz / U_in
    voxel[3] /= U_in ** 2   # p_c / U_in²
    voxel[4] /= U_in ** 2   # dp_dx / U_in²
    voxel[5] /= U_in        # vort_mag / U_in

    for ch in range(6):
        voxel[ch] = (voxel[ch] - means[ch]) / (stds[ch] + 1e-8)

    vt = torch.from_numpy(voxel).unsqueeze(0)  # (1, 6, Nx, Ny, Nz)
    y_norm = np.clip(_tta_predict(model, vt, device), 0.0, 1.0)
    return float(math.exp(y_norm * (_RE_LOG_MAX - _RE_LOG_MIN) + _RE_LOG_MIN))


def predict(ux, uy, uz, p, x_u, y_u, z_u, U_in, fluid_mask, device) -> float:
    """
    Predict Re from a 3D flow field.

    Parameters
    ----------
    ux, uy, uz : np.ndarray (Nx, Ny, Nz)  velocity components in m/s
    p          : np.ndarray (Nx, Ny, Nz)  pressure in Pa (or U_in² units)
    x_u, y_u, z_u : 1-D coordinate arrays (m)
    U_in       : float  inlet velocity (m/s)
    fluid_mask : bool np.ndarray (Nx, Ny, Nz)  True = active fluid voxel
    device     : str  'cpu' or 'cuda'

    Returns
    -------
    float  predicted Reynolds number
    """
    model, stats = load_model(device)

    # Grid spacings
    dx = float(x_u[1] - x_u[0])
    dy = float(y_u[1] - y_u[0])
    dz = float(z_u[1] - z_u[0])

    # Centre pressure over fluid voxels
    p_c = p - p[fluid_mask].mean()

    # Build 6 channels in physical units
    omega_x = _central_diff(uz,  1, dy) - _central_diff(uy,  2, dz)
    omega_y = _central_diff(ux,  2, dz) - _central_diff(uz,  0, dx)
    omega_z = _central_diff(uy,  0, dx) - _central_diff(ux,  1, dy)
    vort_mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)

    ch = np.stack([
        ux,                             # ch0
        uy,                             # ch1
        uz,                             # ch2
        p_c,                            # ch3
        _central_diff(p_c, 0, dx),      # ch4  axial pressure gradient
        vort_mag,                       # ch5  vorticity magnitude
    ], axis=0).astype(np.float32)       # (6, Nx, Ny, Nz)

    # Make dimensionless
    ch[0:3] /= U_in          # velocities / U_in
    ch[3:5] /= (U_in ** 2)   # pressure terms / U_in²
    ch[5]   /= U_in           # vorticity / U_in

    # Zero non-fluid voxels
    ch[:, ~fluid_mask] = 0.0

    # Z-score per channel using training statistics
    means = stats["mean"]   # list of 6 floats
    stds  = stats["std"]    # list of 6 floats
    for i in range(6):
        ch[i] = (ch[i] - means[i]) / (stds[i] + 1e-8)

    # Inference with D4 TTA (2 augmentations: identity + y-flip, averaged)
    tensor = torch.from_numpy(ch).unsqueeze(0)   # (1, 6, Nx, Ny, Nz)
    y_norm = np.clip(_tta_predict(model, tensor, device), 0.0, 1.0)

    # Denormalize log-Re
    Re = math.exp(y_norm * (_RE_LOG_MAX - _RE_LOG_MIN) + _RE_LOG_MIN)
    return float(Re)
