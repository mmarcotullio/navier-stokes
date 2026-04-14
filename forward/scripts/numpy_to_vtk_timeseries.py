#!/usr/bin/env python3
"""
Convert steady-state CFD NumPy files (coords, u, p) into a single VTK
point-cloud file for visualization in ParaView.

Expected input (in --case_dir):

    coords.npy  : (N, 3)
    u.npy       : (N, 3)
    p.npy       : (N,)

Output:

    A single legacy VTK POLYDATA file in --out_dir:

        field_steady.vtk

The file contains:
    - POINTS (coords)
    - POINT_DATA:
        VECTORS velocity
        SCALARS pressure

Optional:
    --mask_pipe and --R_pipe let you mask out points outside a cylinder
    of radius R_pipe in the y–z plane (useful for full Cartesian-box grids).
"""

import os
import argparse
import numpy as np


def write_vtk_points(filename: str,
                     coords_xyz: np.ndarray,
                     u_vec: np.ndarray,
                     p_scalar: np.ndarray):
    """
    Write a legacy VTK POLYDATA file with:
      - POINTS = coords_xyz (N, 3)
      - POINT_DATA:
          VECTORS velocity  (N, 3)
          SCALARS pressure  (N,)
    """
    coords_xyz = np.asarray(coords_xyz, dtype=float)
    u_vec = np.asarray(u_vec, dtype=float)
    p_scalar = np.asarray(p_scalar, dtype=float).reshape(-1)

    assert coords_xyz.shape[0] == u_vec.shape[0] == p_scalar.shape[0], \
        (f"coords ({coords_xyz.shape[0]}), u ({u_vec.shape[0]}), "
         f"p ({p_scalar.shape[0]}) must have the same length")

    N = coords_xyz.shape[0]

    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Steady-state CFD field (NumPy -> VTK)\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        f.write(f"POINTS {N} float\n")
        for i in range(N):
            x, y, z = coords_xyz[i]
            f.write(f"{x:.9e} {y:.9e} {z:.9e}\n")

        f.write(f"VERTICES {N} {N * 2}\n")
        for i in range(N):
            f.write(f"1 {i}\n")

        f.write(f"POINT_DATA {N}\n")

        f.write("VECTORS velocity float\n")
        for i in range(N):
            ux, uy, uz = u_vec[i]
            f.write(f"{ux:.9e} {uy:.9e} {uz:.9e}\n")

        f.write("SCALARS pressure float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for i in range(N):
            f.write(f"{p_scalar[i]:.9e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert steady-state NumPy CFD case (coords, u, p) into a VTK file."
    )
    parser.add_argument(
        "--case_dir",
        required=True,
        help="Directory containing coords.npy, u.npy, p.npy",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for the VTK file (default: <case_dir>/vtk)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="field",
        help="Prefix for the VTK filename (default: 'field')",
    )
    parser.add_argument(
        "--mask_pipe",
        action="store_true",
        help=(
            "If set, mask out points with y^2 + z^2 > R_pipe^2 before writing. "
            "Useful when coords form a full Cartesian box."
        ),
    )
    parser.add_argument(
        "--R_pipe",
        type=float,
        default=0.5,
        help="Pipe radius for --mask_pipe (default: 0.5).",
    )
    args = parser.parse_args()

    case_dir = os.path.abspath(args.case_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(case_dir, "vtk")
    os.makedirs(out_dir, exist_ok=True)

    coords_path = os.path.join(case_dir, "coords.npy")
    u_path = os.path.join(case_dir, "u.npy")
    p_path = os.path.join(case_dir, "p.npy")

    for path in (coords_path, u_path, p_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing file: {path}")

    coords = np.load(coords_path)   # (N, 3)
    u = np.load(u_path)             # (N, 3)
    p = np.load(p_path)             # (N,)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords.npy must have shape (N, 3), got {coords.shape}")
    if u.ndim != 2 or u.shape[1] != 3:
        raise ValueError(f"u.npy must have shape (N, 3), got {u.shape}")
    if p.ndim != 1:
        raise ValueError(f"p.npy must have shape (N,), got {p.shape}")

    N = coords.shape[0]
    if u.shape[0] != N or p.shape[0] != N:
        raise ValueError(
            f"Mismatch: coords N={N}, u N={u.shape[0]}, p N={p.shape[0]}"
        )

    print(f"Loaded:")
    print(f"  coords: {coords.shape}")
    print(f"  u:      {u.shape}")
    print(f"  p:      {p.shape}")

    if args.mask_pipe:
        y = coords[:, 1]
        z = coords[:, 2]
        R = float(args.R_pipe)
        mask = (y * y + z * z) <= R * R + 1e-12
        n_kept = int(mask.sum())

        if n_kept == 0:
            raise RuntimeError(
                f"--mask_pipe removed all points (R_pipe={R}). "
                "Check that coords are in physical units."
            )

        print(f"Pipe mask R={R:.4f}: keeping {n_kept}/{N} points "
              f"({100.0 * n_kept / N:.1f}%).")
        coords = coords[mask]
        u = u[mask]
        p = p[mask]

    filename = os.path.join(out_dir, f"{args.prefix}_steady.vtk")
    write_vtk_points(filename, coords, u, p)
    print(f"Wrote: {filename}")


if __name__ == "__main__":
    main()
