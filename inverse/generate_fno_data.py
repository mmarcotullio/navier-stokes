"""
Generate synthetic training data for CNN3DInverse by querying the trained FNO
forward model at a dense grid of (Re, U_in) parameter combinations.

Resolution note
---------------
The FNO's spectral content is limited to (modes_x=16, modes_y=10, modes_z=10).
Querying at a resolution higher than the training grid (265×20×20) would only
interpolate the same low-frequency field — it adds no new physical information.
We therefore query at the native training resolution (265×20×20).

Output layout
-------------
    inverse/fno_training_data/
        metadata.csv            — case_id, Re, U_in
        case_00000/
            voxel_grid_raw.npy  — (6, 265, 20, 20) float32 raw voxel
        case_00001/
            ...

Voxel channels (raw, not divided by U_in, not z-scored):
    [ux, uy, uz, p_centered, dp_dx, |ω|]

Same format as precompute_voxels.py — dataset.py's fast path loads this directly.
With use_true_uin=True, train.py divides by the exact U_in from metadata.csv,
recovering the dimensionless FNO output with no Stage-1 error.

Run from the inverse/ directory:
    python generate_fno_data.py
"""

import csv
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# Add forward/fno_model to path for FNO3d
_FWD_MODEL = os.path.join(_ROOT, "forward", "fno_model")
for _p in [_FWD_MODEL, _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fno_model import FNO3d                          # noqa: E402
from dataset import _central_diff                    # noqa: E402
from utils import CYL_R, CYL_X, PIPE_L, PIPE_R      # noqa: E402
from utils import RE_MIN, RE_MAX, UIN_MIN, UIN_MAX   # noqa: E402

# ── Configuration ─────────────────────────────────────────────────────────────

FNO_CKPT = os.path.join(_ROOT, "models", "fno3d_best_forward.pt")
OUT_DIR  = os.path.join(_HERE, "fno_training_data")

# Grid dimensions — match FNO training resolution
NX, NY, NZ = 265, 20, 20

# Sampling: 1000 Re values (log-uniform) × 3 U_in values = 3,000 samples
# ~7.3 GB storage; 2,700 train × 8 D4 aug = 21,600 effective samples/epoch
N_RE     = 1000
UIN_VALS = [0.1, 0.55, 1.0]

BATCH_SIZE = 8


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pad_to_multiple4(x: torch.Tensor):
    """Pad spatial dims to next multiple of 4 for FFT efficiency.

    Returns (padded_tensor, Nx_orig, Ny_orig, Nz_orig) so the caller can
    crop back after inference.
    """
    import torch.nn.functional as F
    _, _, Nx, Ny, Nz = x.shape
    Nx_p = ((Nx + 3) // 4) * 4
    Ny_p = ((Ny + 3) // 4) * 4
    Nz_p = ((Nz + 3) // 4) * 4
    if (Nx_p, Ny_p, Nz_p) == (Nx, Ny, Nz):
        return x, Nx, Ny, Nz
    x = F.pad(x, (0, Nz_p - Nz, 0, Ny_p - Ny, 0, Nx_p - Nx))
    return x, Nx, Ny, Nz


def build_geometry():
    """Build fixed geometry arrays for the 265×20×20 Cartesian grid.

    Coordinate convention matches fno_dataset.py:
        x_norm, y_norm, z_norm each normalised to [0, 1] per-axis.
    """
    xs = np.linspace(0.0,    PIPE_L, NX, dtype=np.float32)   # [0, 10] m
    ys = np.linspace(-PIPE_R, PIPE_R, NY, dtype=np.float32)   # [-0.5, 0.5] m
    zs = np.linspace(-PIPE_R, PIPE_R, NZ, dtype=np.float32)   # [-0.5, 0.5] m

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")   # (265, 20, 20)

    x_norm = (X - xs.min()) / (xs.max() - xs.min())
    y_norm = (Y - ys.min()) / (ys.max() - ys.min())
    z_norm = (Z - zs.min()) / (zs.max() - zs.min())

    R2D         = np.sqrt(Y ** 2 + Z ** 2)
    dist_sphere = np.sqrt((X - CYL_X) ** 2 + Y ** 2 + Z ** 2)
    fluid_mask  = (R2D <= PIPE_R).astype(np.float32)         # inside pipe
    cyl_mask    = (dist_sphere <= CYL_R).astype(np.float32)  # inside sphere

    # Full fluid domain used to zero non-fluid voxels in the output
    full_fluid = (R2D <= PIPE_R) & (dist_sphere > CYL_R)

    # Fixed 5-channel geometry part of FNO input
    geom_np = np.stack([x_norm, y_norm, z_norm, fluid_mask, cyl_mask])
    # (5, 265, 20, 20)

    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    dz = float(zs[1] - zs[0])

    return geom_np, full_fluid, dx, dy, dz


def fno_to_voxel(
    fno_out_np: np.ndarray,
    U_in: float,
    full_fluid: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """Convert a single FNO output (4, Nx, Ny, Nz) to a raw 6-channel voxel.

    FNO output channels: [ux/U_in, uy/U_in, uz/U_in, p/U_in²]
    Voxel channels:      [ux, uy, uz, p_centered, dp_dx, |ω|]

    p is already fluid-mean-centred by the FNO training pipeline.
    Non-fluid voxels are zeroed.
    """
    ux  = fno_out_np[0] * U_in               # (Nx, Ny, Nz)
    uy  = fno_out_np[1] * U_in
    uz  = fno_out_np[2] * U_in
    p_c = fno_out_np[3] * (U_in ** 2)        # already mean-centred

    # Axial pressure gradient dp/dx (2nd-order FD)
    dp_dx = _central_diff(p_c, ax=0, h=dx)

    # Vorticity magnitude |curl(u)|
    omega_x  = _central_diff(uz, 1, dy) - _central_diff(uy, 2, dz)
    omega_y  = _central_diff(ux, 2, dz) - _central_diff(uz, 0, dx)
    omega_z  = _central_diff(uy, 0, dx) - _central_diff(ux, 1, dy)
    vort_mag = np.sqrt(omega_x ** 2 + omega_y ** 2 + omega_z ** 2)

    voxel = np.stack([ux, uy, uz, p_c, dp_dx, vort_mag]).astype(np.float32)
    voxel[:, ~full_fluid] = 0.0

    return voxel


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device  : {device}")
    print(f"FNO ckpt: {FNO_CKPT}")
    if not os.path.isfile(FNO_CKPT):
        raise FileNotFoundError(f"FNO checkpoint not found: {FNO_CKPT}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load FNO ──────────────────────────────────────────────────────────────
    model = FNO3d(
        modes_x=16, modes_y=10, modes_z=10,
        width=32, in_channels=7, out_channels=4, n_layers=4,
    )
    model.load_state_dict(
        torch.load(FNO_CKPT, map_location=device, weights_only=True)
    )
    model.to(device).eval()
    print("FNO loaded.\n")

    # ── Geometry (fixed for all samples) ──────────────────────────────────────
    geom_np, full_fluid, dx, dy, dz = build_geometry()
    # geom_np : (5, 265, 20, 20) float32

    # ── (Re, U_in) pair list ──────────────────────────────────────────────────
    Re_vals   = np.exp(
        np.linspace(np.log(RE_MIN), np.log(RE_MAX), N_RE)
    ).tolist()
    all_pairs = [(re, uin) for re in Re_vals for uin in UIN_VALS]
    N_TOTAL   = len(all_pairs)
    print(f"Sampling: {N_RE} Re (log-uniform) × {len(UIN_VALS)} U_in = {N_TOTAL} total")
    print(f"Output  : {OUT_DIR}")
    print(f"Storage : ~{N_TOTAL * 2.43 / 1024:.1f} GB\n")

    # ── Generate ──────────────────────────────────────────────────────────────
    meta_path = os.path.join(OUT_DIR, "metadata.csv")
    with open(meta_path, "w", newline="") as meta_f:
        writer = csv.writer(meta_f)
        writer.writerow(["case_id", "Re", "U_in"])

        n_batches = (N_TOTAL + BATCH_SIZE - 1) // BATCH_SIZE

        for b_idx in range(n_batches):
            batch = all_pairs[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
            B     = len(batch)

            # Build (B, 7, 265, 20, 20) input on CPU, then move to device
            inp_np = np.zeros((B, 7, NX, NY, NZ), dtype=np.float32)
            for i, (re, uin) in enumerate(batch):
                inp_np[i, :5] = geom_np
                inp_np[i,  5] = (re  - RE_MIN)  / (RE_MAX  - RE_MIN)   # Re_norm
                inp_np[i,  6] = (uin - UIN_MIN) / (UIN_MAX - UIN_MIN)  # U_in_norm

            inp_t = torch.from_numpy(inp_np).to(device)

            # Pad to multiple-of-4 for FFT efficiency
            inp_padded, Nx_orig, Ny_orig, Nz_orig = _pad_to_multiple4(inp_t)

            with torch.no_grad():
                out_padded = model(inp_padded)   # (B, 4, Nx_p, Ny_p, Nz_p)

            out_np = (
                out_padded[:, :, :Nx_orig, :Ny_orig, :Nz_orig]
                .cpu().numpy()
            )   # (B, 4, 265, 20, 20)

            for local_i, (re, uin) in enumerate(batch):
                case_idx = b_idx * BATCH_SIZE + local_i
                case_id  = f"case_{case_idx:05d}"
                case_dir = os.path.join(OUT_DIR, case_id)
                os.makedirs(case_dir, exist_ok=True)

                voxel = fno_to_voxel(
                    out_np[local_i], uin, full_fluid, dx, dy, dz
                )
                np.save(os.path.join(case_dir, "voxel_grid_raw.npy"), voxel)
                writer.writerow([case_id, f"{re:.6f}", f"{uin:.4f}"])

            # Progress
            done = min((b_idx + 1) * BATCH_SIZE, N_TOTAL)
            pct  = done / N_TOTAL * 100
            print(f"\r  {done:5d}/{N_TOTAL}  ({pct:5.1f}%)", end="", flush=True)

    print(f"\n\nDone. {N_TOTAL} samples written to {OUT_DIR}/")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
