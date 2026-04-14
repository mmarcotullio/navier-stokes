"""
VoxelCFDDataset — steady-state CFD data for the 3D CNN inverse surrogate.

Loads the structured Cartesian grid (from fno_grid/) as a (6, Nx, Ny, Nz)
voxel tensor. Coordinates are implicit in the grid position — no xyz channels.

Input channels (dimensionless, z-scored)
-----------------------------------------
  0 : ux / U_in_pred                 (dimensionless axial velocity)
  1 : uy / U_in_pred                 (dimensionless lateral velocity y)
  2 : uz / U_in_pred                 (dimensionless lateral velocity z)
  3 : p_centered / U_in_pred²        (dimensionless pressure)
  4 : (dp_centered/dx) / U_in_pred²  (axial pressure gradient, D=1m implicit)
  5 : |ω| / U_in_pred                (vorticity magnitude, D=1m implicit)

Non-fluid voxels (outside pipe, inside sphere, unsampled) are zeroed.

Fast path (cache exists — voxel_grid_raw.npy written by precompute_voxels.py):
    Load (6, Nx, Ny, Nz) float32 array of raw values:
        [ux, uy, uz, p_centered, dp_dx, |ω|]
    Apply U_in division, z-score, and D4 augmentation in __getitem__.

Slow path (no cache — used when precompute_voxels.py has not been run):
    Load coords.npy (N,3), u.npy (N,3), p.npy (N,), sampled_mask.npy (N,).
    Sort to C-order (x-major), reshape to (Nx, Ny, Nz).
    Build fluid mask, compute derived channels dp/dx and |ω|, stack.

Augmentation (training only) — D4 dihedral group (8 variants)
--------------------------------------------------------------
4 rotations × 2 reflection states around the pipe (x) axis. All 8 are
exact physical symmetries of the pipe+centred-sphere geometry.
  - flip: y → −y  (torch.flip on Ny axis + negate channel 1/uy)
  - rotation: k×90° CCW (torch.rot90 on Ny/Nz dims + rotate channels 1/2)
  - Gaussian noise on all 6 channels (after z-score)

Labels
------
    y = FloatTensor (1,) = [Re_log_norm]
"""

import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import CYL_R, CYL_X, PIPE_L, PIPE_R, normalize_re

# Upstream fraction threshold: x < 30% of domain = pre-obstacle region
_UPSTREAM_X_FRAC = CYL_X / PIPE_L   # 0.3


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
        # Only two points: simple first-order
        out[_sl(ax, 0)]  = (arr[_sl(ax, 1)] - arr[_sl(ax, 0)]) / h
        out[_sl(ax, -1)] = out[_sl(ax, 0)]
    else:
        # Single point: gradient is zero
        out[:] = 0.0

    return out


# ── Dataset class ─────────────────────────────────────────────────────────────

class VoxelCFDDataset(Dataset):
    """
    Voxel-grid dataset for CNN inverse regression.

    One sample per CFD case.

    Args:
        root_dir     : path to fno_grid/ directory (contains metadata.csv
                       and case_XXXX/ sub-directories).
        metadata_csv : CSV filename relative to root_dir (default "metadata.csv").
        augment      : D4 dihedral augmentation — 4 rotations × 2 reflections
                       around the x-axis (default False).
        noise_std    : std of Gaussian noise added to all 6 channels after
                       z-score normalisation (default 0.0).
        field_stats  : {"mean": [6], "std": [6]} for z-score normalisation.
                       If None, no normalisation is applied.
        uin_linear   : {"a": float, "b": float} — linear U_in predictor.
                       Required for __getitem__; ValueError raised if None.
    """

    def __init__(
        self,
        root_dir:      str,
        metadata_csv:  str           = "metadata.csv",
        augment:       bool          = False,
        noise_std:     float         = 0.0,
        field_stats:   "dict | None" = None,
        uin_linear:    "dict | None" = None,
        use_true_uin:  bool          = False,
    ):
        super().__init__()
        self.root_dir     = root_dir
        self.augment      = augment
        self.noise_std    = noise_std
        self.field_stats  = field_stats
        self.uin_linear   = uin_linear
        self.use_true_uin = use_true_uin

        meta_path = os.path.join(root_dir, metadata_csv)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata CSV not found: {meta_path}")

        self.cases: list = []
        with open(meta_path, newline="") as f:
            for row in csv.DictReader(f):
                self.cases.append(row)

        # Only include cases whose on-disk data exists.
        # Accept either:
        #   (a) voxel_grid_raw.npy alone  (FNO-generated data)
        #   (b) u.npy + p.npy             (raw CFD data, slow-path fallback)
        self.sample_index: list = []
        for ci, row in enumerate(self.cases):
            case_dir = os.path.join(root_dir, row["case_id"])
            if not os.path.isdir(case_dir):
                continue
            has_cache = os.path.isfile(
                os.path.join(case_dir, "voxel_grid_raw.npy")
            )
            has_raw = os.path.isfile(
                os.path.join(case_dir, "u.npy")
            ) and os.path.isfile(os.path.join(case_dir, "p.npy"))
            if has_cache or has_raw:
                self.sample_index.append(ci)

    def __len__(self) -> int:
        return len(self.sample_index)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_raw(self, ci: int):
        """Load coords, u, p, sampled_mask for case index ci."""
        row = self.cases[ci]
        d   = os.path.join(self.root_dir, row["case_id"])
        coords = np.load(os.path.join(d, "coords.npy")).astype(np.float64)
        u      = np.load(os.path.join(d, "u.npy")).astype(np.float64)
        p      = np.load(os.path.join(d, "p.npy")).astype(np.float64)
        sm_path = os.path.join(d, "sampled_mask.npy")
        sm = (
            np.load(sm_path).astype(bool)
            if os.path.isfile(sm_path)
            else np.ones(len(coords), dtype=bool)
        )
        return row, coords, u, p, sm

    @staticmethod
    def _build_voxel_raw(
        coords: np.ndarray,
        u:      np.ndarray,
        p:      np.ndarray,
        sm:     np.ndarray,
    ) -> "tuple[np.ndarray, np.ndarray]":
        """
        Convert unordered CFD grid arrays to a (6, Nx, Ny, Nz) raw voxel tensor.

        Channels: [ux, uy, uz, p_centered, dp_dx, |omega|]
        Non-fluid voxels are zeroed.

        Returns:
            voxel   : (6, Nx, Ny, Nz) float32
            fluid3d : (Nx, Ny, Nz) bool fluid mask
        """
        # ── Infer grid shape ──────────────────────────────────────────────────
        ux_vals = np.unique(np.round(coords[:, 0], 6))
        uy_vals = np.unique(np.round(coords[:, 1], 6))
        uz_vals = np.unique(np.round(coords[:, 2], 6))
        Nx, Ny, Nz = len(ux_vals), len(uy_vals), len(uz_vals)

        # ── Sort to C-order: x-major (slowest), z-fastest ────────────────────
        sort_idx = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
        u3d  = u[sort_idx].reshape(Nx, Ny, Nz, 3)
        p3d  = p[sort_idx].reshape(Nx, Ny, Nz)
        sm3d = sm[sort_idx].reshape(Nx, Ny, Nz)
        x3d  = coords[sort_idx, 0].reshape(Nx, Ny, Nz)
        y3d  = coords[sort_idx, 1].reshape(Nx, Ny, Nz)
        z3d  = coords[sort_idx, 2].reshape(Nx, Ny, Nz)

        # ── Fluid mask: inside pipe, outside sphere, sampled ──────────────────
        r3d     = np.sqrt(y3d ** 2 + z3d ** 2)
        dist_sp = np.sqrt((x3d - CYL_X) ** 2 + y3d ** 2 + z3d ** 2)
        fluid3d = (r3d <= PIPE_R) & (dist_sp > CYL_R) & sm3d

        # ── Pressure mean-centering ───────────────────────────────────────────
        p_mean = p3d[fluid3d].mean() if fluid3d.any() else p3d.mean()
        p_c3d  = p3d - p_mean

        # ── Grid spacings for finite differences ──────────────────────────────
        dx = float(ux_vals[1] - ux_vals[0]) if Nx > 1 else 1.0
        dy = float(uy_vals[1] - uy_vals[0]) if Ny > 1 else 1.0
        dz = float(uz_vals[1] - uz_vals[0]) if Nz > 1 else 1.0

        # ── Axial pressure gradient dp/dx ─────────────────────────────────────
        dp_dx3d = _central_diff(p_c3d, ax=0, h=dx)

        # ── Vorticity magnitude |ω| = |curl(u)| ───────────────────────────────
        # omega_x = duz/dy − duy/dz
        # omega_y = dux/dz − duz/dx
        # omega_z = duy/dx − dux/dy
        ux3d = u3d[:, :, :, 0]
        uy3d = u3d[:, :, :, 1]
        uz3d = u3d[:, :, :, 2]

        omega_x = _central_diff(uz3d, 1, dy) - _central_diff(uy3d, 2, dz)
        omega_y = _central_diff(ux3d, 2, dz) - _central_diff(uz3d, 0, dx)
        omega_z = _central_diff(uy3d, 0, dx) - _central_diff(ux3d, 1, dy)
        omega_mag = np.sqrt(omega_x ** 2 + omega_y ** 2 + omega_z ** 2)

        # ── Stack (6, Nx, Ny, Nz) and zero non-fluid ─────────────────────────
        voxel = np.stack(
            [ux3d, uy3d, uz3d, p_c3d, dp_dx3d, omega_mag],
        ).astype(np.float32)           # (6, Nx, Ny, Nz)
        voxel[:, ~fluid3d] = 0.0

        return voxel, fluid3d

    # ── Dataset API ───────────────────────────────────────────────────────────

    def __getitem__(self, idx: int):
        if self.uin_linear is None and not self.use_true_uin:
            raise ValueError(
                "uin_linear must be provided (or use_true_uin=True). "
                "VoxelCFDDataset requires a fitted linear U_in model or "
                "the true U_in from metadata to build dimensionless channels."
            )

        ci  = self.sample_index[idx]
        row = self.cases[ci]
        d   = os.path.join(self.root_dir, row["case_id"])
        Re  = float(row["Re"])

        # ── Load voxel grid (fast or slow path) ───────────────────────────────
        cache_path = os.path.join(d, "voxel_grid_raw.npy")
        if os.path.isfile(cache_path):
            voxel_raw = np.load(cache_path).astype(np.float32)  # (6, Nx, Ny, Nz)
        else:
            _, coords, u, p, sm = self._load_raw(ci)
            voxel_raw, _ = self._build_voxel_raw(coords, u, p, sm)

        _, Nx, Ny, Nz = voxel_raw.shape

        # ── Upstream ux mean (non-zero fluid voxels only) ─────────────────────
        n_up = max(1, int(_UPSTREAM_X_FRAC * Nx))
        up_ux = voxel_raw[0, :n_up, :, :]
        nz    = up_ux != 0.0
        ux_mean_raw = float(up_ux[nz].mean()) if nz.any() else float(up_ux.mean())

        # ── Stage 1: predict U_in (or read true value from metadata) ─────────
        if self.use_true_uin:
            U_pred = float(row["U_in"])
        else:
            a_lin  = self.uin_linear["a"]
            b_lin  = self.uin_linear["b"]
            U_pred = float(np.clip(a_lin * ux_mean_raw + b_lin, 0.05, 1.5))

        # ── Build dimensionless channels ──────────────────────────────────────
        voxel = voxel_raw.copy()
        voxel[0] /= U_pred          # ux / U
        voxel[1] /= U_pred          # uy / U
        voxel[2] /= U_pred          # uz / U
        voxel[3] /= U_pred ** 2     # p_centered / U²    [kinematic: m²/s² / m²/s²]
        voxel[4] /= U_pred ** 2     # dp/dx × D / U²     [D=1m, m/s² / m²/s² = 1/m → ×1m = dim.less]
        voxel[5] /= U_pred          # |ω| × D / U        [D=1m, 1/s × 1m / m/s = dim.less]

        # ── Z-score normalisation ─────────────────────────────────────────────
        if self.field_stats is not None:
            m = self.field_stats["mean"]
            s = self.field_stats["std"]
            for ch in range(6):
                voxel[ch] = (voxel[ch] - m[ch]) / (s[ch] + 1e-8)

        # ── Convert to tensor ─────────────────────────────────────────────────
        vt = torch.from_numpy(voxel)   # (6, Nx, Ny, Nz)

        # ── D4 augmentation (training only) ───────────────────────────────────
        if self.augment:
            k    = int(np.random.randint(0, 4))   # rotation: 0/90/180/270° CCW
            flip = int(np.random.randint(0, 2))   # 0 = keep, 1 = y→(−y) reflect

            if flip:
                # Reflect y-axis spatially; negate uy (vector component)
                # dp/dx and |ω| are scalars — no sign change needed
                vt    = torch.flip(vt, dims=[2])
                vt[1] = -vt[1]

            if k > 0:
                # Rotate (Ny, Nz) plane k×90°; rotate (uy, uz) velocity vector
                vt = torch.rot90(vt, k=k, dims=[2, 3])
                for _ in range(k):
                    uy_tmp = vt[1].clone()
                    vt[1]  = -vt[2]
                    vt[2]  =  uy_tmp

            if self.noise_std > 0.0:
                vt = vt + self.noise_std * torch.randn_like(vt)

        y = torch.tensor([normalize_re(Re)], dtype=torch.float32)
        return vt, y

    def get_ux_mean_upstream(self, idx: int) -> float:
        """
        Return raw (un-normalised) mean ux in the upstream region for a case.

        Fast when voxel_grid_raw.npy cache exists; falls back to raw-grid load.
        Used by train.py to compute Stage 1 U_in predictions without building
        the full 6-channel voxel tensor.
        """
        ci  = self.sample_index[idx]
        row = self.cases[ci]
        d   = os.path.join(self.root_dir, row["case_id"])

        cache_path = os.path.join(d, "voxel_grid_raw.npy")
        if os.path.isfile(cache_path):
            # Memory-mapped read of channel 0 (ux) only
            vox  = np.load(cache_path, mmap_mode="r")
            Nx   = vox.shape[1]
            n_up = max(1, int(_UPSTREAM_X_FRAC * Nx))
            up   = np.asarray(vox[0, :n_up, :, :])
            nz   = up != 0.0
            return float(up[nz].mean()) if nz.any() else float(up.mean())

        # Slow path: load raw grid and extract upstream ux
        _, coords, u, p, sm = self._load_raw(ci)
        r       = np.sqrt(coords[:, 1] ** 2 + coords[:, 2] ** 2)
        dist_sp = np.sqrt(
            (coords[:, 0] - CYL_X) ** 2 + coords[:, 1] ** 2 + coords[:, 2] ** 2
        )
        fluid = (r <= PIPE_R) & (dist_sp > CYL_R) & sm
        if not fluid.any():
            return 0.0
        lo   = coords.min(axis=0)
        hi   = coords.max(axis=0)
        span = np.where(hi - lo > 1e-8, hi - lo, 1.0)
        x_nm = (coords[fluid, 0] - lo[0]) / span[0]
        up   = x_nm < _UPSTREAM_X_FRAC
        ux   = u[fluid, 0]
        return float(ux[up].mean()) if up.any() else float(ux.mean())

    # ── Field statistics computation ──────────────────────────────────────────

    @staticmethod
    def compute_dimless_stats(
        root_dir:       str,
        metadata_csv:   str,
        train_case_cis: list,
        uin_linear:     dict,
        use_true_uin:   bool = False,
    ) -> dict:
        """
        Compute per-channel mean and std for all 6 dimensionless channels
        over training cases (fluid voxels only — non-zero mask).

        Must be called after Stage 1 linear fit.

        Returns:
            {"mean": [6], "std": [6]}
        """
        meta_path = os.path.join(root_dir, metadata_csv)
        cases: list = []
        with open(meta_path, newline="") as f:
            for row in csv.DictReader(f):
                cases.append(row)

        a_lin = uin_linear.get("a", 0.0)
        b_lin = uin_linear.get("b", 0.0)

        counts  = np.zeros(6, dtype=np.float64)
        sums    = np.zeros(6, dtype=np.float64)
        sum_sqs = np.zeros(6, dtype=np.float64)

        # Divisors matching __getitem__: ch 0,1,2,5 → /U; ch 3,4 → /U²
        div_pow = [1, 1, 1, 2, 2, 1]

        for ci in train_case_cis:
            row = cases[ci]
            d   = os.path.join(root_dir, row["case_id"])
            if not os.path.isdir(d):
                continue

            cache_path = os.path.join(d, "voxel_grid_raw.npy")
            if os.path.isfile(cache_path):
                vox = np.load(cache_path).astype(np.float64)   # (6, Nx, Ny, Nz)
            else:
                c_p = os.path.join(d, "coords.npy")
                u_p = os.path.join(d, "u.npy")
                p_p = os.path.join(d, "p.npy")
                if not (os.path.isfile(c_p) and os.path.isfile(u_p) and os.path.isfile(p_p)):
                    continue
                coords = np.load(c_p).astype(np.float64)
                u_arr  = np.load(u_p).astype(np.float64)
                p_arr  = np.load(p_p).astype(np.float64)
                sm_path = os.path.join(d, "sampled_mask.npy")
                sm = (
                    np.load(sm_path).astype(bool)
                    if os.path.isfile(sm_path)
                    else np.ones(len(coords), dtype=bool)
                )
                vox_f32, _ = VoxelCFDDataset._build_voxel_raw(coords, u_arr, p_arr, sm)
                vox = vox_f32.astype(np.float64)

            if use_true_uin:
                U_pred = float(row["U_in"])
            else:
                # Upstream ux mean (non-zero proxy for fluid mask)
                Nx   = vox.shape[1]
                n_up = max(1, int(_UPSTREAM_X_FRAC * Nx))
                up   = vox[0, :n_up, :, :]
                nz   = up != 0.0
                ux_mean = float(up[nz].mean()) if nz.any() else float(up.mean())
                U_pred = float(np.clip(a_lin * ux_mean + b_lin, 0.05, 1.5))

            # Fluid mask: at least one channel non-zero
            fluid_mask = np.zeros(vox.shape[1:], dtype=bool)
            for ch in range(6):
                fluid_mask |= (vox[ch] != 0.0)

            for ch in range(6):
                vals = vox[ch][fluid_mask] / (U_pred ** div_pow[ch])
                counts[ch]  += len(vals)
                sums[ch]    += vals.sum()
                sum_sqs[ch] += (vals ** 2).sum()

        n     = counts.clip(min=1)
        means = sums / n
        stds  = np.sqrt((sum_sqs / n - means ** 2).clip(min=0.0))
        return {"mean": means.tolist(), "std": stds.tolist()}
