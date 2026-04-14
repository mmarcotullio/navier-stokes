"""
CNN3DInverse — 3D ResNet with SE channel attention for CFD inverse surrogate.

Architecture
------------
Predicts Re_log_norm ∈ [0, 1] from a voxelised 3D dimensionless field tensor.

Input:  (B, in_channels, Nx, Ny, Nz)
Output: (B, 1)  — Re_log_norm  [clamp to [0, 1] at inference]

The pipe domain has a 13:1 axial-to-lateral aspect ratio (~265 vs 20 voxels).
Standard 3×3×3 kernels grow the axial receptive field by only 2 voxels per
block — after 5 blocks the RF spans 11 axial voxels, far too small to cover
the sphere wake (up to ~80 voxels at native resolution).

Anisotropic (k_ax, 3, 3) kernels with k_ax=5 in blocks 2–5 grow the axial RF
by 4 voxels per block. After 5 blocks the RF reaches 182 axial voxels —
covering the full recirculation zone for all Re ∈ [100, 1000].

SE (Squeeze-and-Excitation) channel attention re-weights the 6 input channels
per layer: dp/dx dominates at low Re, |ω| at high Re, ux/U in the profile
region. SE learns this weighting with negligible parameter overhead.

Residual connections prevent training stalls on the small dataset (900 cases)
by providing a gradient highway through all 5 blocks.

Axial receptive field by block (input Nx=265):
    Block 1 (k_ax=3, no pool)       : RF =   3
    Block 2 (k_ax=5, MaxPool(2,2,2)): RF =  14
    Block 3 (k_ax=5, MaxPool(2,2,2)): RF =  38
    Block 4 (k_ax=5, MaxPool(2,1,1)): RF =  86
    Block 5 (k_ax=5, MaxPool(2,1,1)): RF = 182

Layer shapes (input (B, 6, 265, 20, 20)):
    Block 1 : (B,  32, 265, 20, 20)
    Block 2 : (B,  64, 132, 10, 10)
    Block 3 : (B, 128,  66,  5,  5)
    Block 4 : (B, 256,  33,  5,  5)
    Block 5 : (B, 256,  16,  5,  5)
    GAP     : (B, 256)
    Head    : (B,   1)

Parameters: ~7.5 M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention (Hu et al. 2018).

    Computes a per-channel scale factor from global average-pooled statistics.
    Adds C²/reduction parameters per block — total overhead ~22 K across all
    5 blocks with reduction=8.
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

    k_ax=5 for blocks 2–5 elongates the axial receptive field to match the
    13:1 aspect ratio of the pipe domain. k_ax=3 for block 1 (no pool, full
    spatial resolution — smaller kernel is cheaper and sufficient early on).

    Pooling is applied AFTER the residual addition, not before. This means
    the skip connection and both conv branches see the same spatial resolution
    and no dimension-matching is needed in the pool step itself.
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
        # 1×1×1 projection when channel count changes; identity otherwise
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

    Args:
        in_channels : number of input field channels (default 6).
                      6 = [ux/U, uy/U, uz/U, p/U², dp/dx/U², |ω|/U]

    Input:  (B, in_channels, Nx, Ny, Nz)
    Output: (B, 1) — Re_log_norm prediction (un-clamped; clamp to [0,1]
            after inference: torch.clamp(model(x), 0.0, 1.0))
    """

    def __init__(self, in_channels: int = 6):
        super().__init__()
        self.blocks = nn.Sequential(
            # Block 1: no pooling — retain full spatial resolution early on
            ResBlock3d(in_channels,  32, k_ax=3, pool=None),
            # Block 2: MaxPool(2,2,2) — axial 265→132, lateral 20→10
            ResBlock3d(32,   64, k_ax=5, pool=nn.MaxPool3d((2, 2, 2))),
            # Block 3: MaxPool(2,2,2) — axial 132→66, lateral 10→5
            ResBlock3d(64,  128, k_ax=5, pool=nn.MaxPool3d((2, 2, 2))),
            # Block 4: MaxPool(2,1,1) — axial-only (lateral 5×5 too small for 2×2)
            ResBlock3d(128, 256, k_ax=5, pool=nn.MaxPool3d((2, 1, 1))),
            # Block 5: MaxPool(2,1,1) — axial-only
            ResBlock3d(256, 256, k_ax=5, pool=nn.MaxPool3d((2, 1, 1))),
        )
        # Collapse remaining spatial volume to a 256-d global descriptor
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
        """
        Args:
            x : (B, in_channels, Nx, Ny, Nz) — dimensionless voxel grid.
                No fixed Nx requirement; AdaptiveAvgPool3d handles any length.

        Returns:
            (B, 1) — Re_log_norm prediction.
        """
        x = self.blocks(x)
        x = self.gap(x).flatten(1)   # (B, 256)
        return self.head(x)           # (B, 1)
