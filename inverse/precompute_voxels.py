"""
precompute_voxels.py — one-time conversion of CFD grid data to voxel tensors.

For each case, loads coords.npy / u.npy / p.npy / sampled_mask.npy from the
fno_grid/ directory, builds the 3D fluid-masked voxel grid, and saves a
compact (6, Nx, Ny, Nz) float32 array as voxel_grid_raw.npy per case.

After running this script, VoxelCFDDataset.__getitem__ uses the fast path
(np.load + U_in division + z-score) instead of the slow path (np.lexsort +
reshape + finite differences on the full grid).

Output per case
---------------
    {case_dir}/voxel_grid_raw.npy  — shape (6, Nx, Ny, Nz), float32
        channels: [ux, uy, uz, p_centered, dp_dx, |omega|]

    Channels are raw (NOT divided by U_in, NOT z-score normalised). That
    happens per-epoch in __getitem__ using stats computed from training only.
    Non-fluid voxels are zeroed.

File size: ~6 × 265 × 20 × 20 × 4 bytes ≈ 2.54 MB per case
           → ~2.5 GB for 1000 cases

Run once before training:
    cd inverse/
    python precompute_voxels.py
"""

import csv
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dataset import VoxelCFDDataset   # noqa: E402

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "cfd_training_data", "fno_grid",
)
META_CSV = "metadata.csv"


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    meta_path = os.path.join(DATA_DIR, META_CSV)
    if not os.path.isfile(meta_path):
        print(f"ERROR: metadata CSV not found: {meta_path}")
        sys.exit(1)

    cases: list = []
    with open(meta_path, newline="") as f:
        for row in csv.DictReader(f):
            cases.append(row)

    print(f"Found {len(cases)} cases in {meta_path}")
    print("Saving (6, Nx, Ny, Nz) voxel_grid_raw.npy per case\n")

    n_ok = n_skip = n_err = 0
    total_mb = 0.0

    for ci, row in enumerate(cases):
        case_id  = row["case_id"]
        case_dir = os.path.join(DATA_DIR, case_id)
        out_path = os.path.join(case_dir, "voxel_grid_raw.npy")

        if not os.path.isdir(case_dir):
            print(f"  {case_id}: SKIP (directory not found)")
            n_skip += 1
            continue

        if os.path.isfile(out_path):
            print(f"  {case_id}: already cached, skipping")
            n_skip += 1
            continue

        try:
            coords = np.load(os.path.join(case_dir, "coords.npy")).astype(np.float64)
            u      = np.load(os.path.join(case_dir, "u.npy")).astype(np.float64)
            p      = np.load(os.path.join(case_dir, "p.npy")).astype(np.float64)

            sm_path = os.path.join(case_dir, "sampled_mask.npy")
            sm = (
                np.load(sm_path).astype(bool)
                if os.path.isfile(sm_path)
                else np.ones(coords.shape[0], dtype=bool)
            )

            voxel, fluid3d = VoxelCFDDataset._build_voxel_raw(coords, u, p, sm)

            if not fluid3d.any():
                print(f"  {case_id}: WARNING — empty fluid domain, skipping")
                n_skip += 1
                continue

            np.save(out_path, voxel)
            mb = voxel.nbytes / 1e6
            total_mb += mb
            shape_str = "×".join(str(s) for s in voxel.shape)
            print(f"  {case_id}: shape ({shape_str}), {mb:.2f} MB")
            n_ok += 1

        except Exception as exc:
            print(f"  {case_id}: ERROR — {exc}")
            n_err += 1

    print(f"\nDone. Cached {n_ok} cases, skipped {n_skip}, errors {n_err}.")
    if n_ok > 0:
        print(f"Total disk usage: {total_mb:.0f} MB")


if __name__ == "__main__":
    main()
