#!/usr/bin/env python3
import os
import numpy as np

# -----------------------
# Pipe geometry (physical units)
# -----------------------
L_PIPE = 10.0   # pipe length
R_PIPE = 0.5    # pipe radius

# -----------------------
# Cross-section resolution (uniform in y and z)
# -----------------------
Ny = 20
Nz = 20

# Choose Nx so spacing in x matches dy (based on y spacing)
dy = (2.0 * R_PIPE) / (Ny - 1)
Nx = int(L_PIPE / dy) + 1
dx = L_PIPE / (Nx - 1)

print(f"Using Nx={Nx}, Ny={Ny}, Nz={Nz}")
print(f"Computed dy = {dy:.6f}")
print(f"Computed dx = {dx:.6f} (dx ≈ dy)")

# -----------------------
# Output path
# -----------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CFD_DIR = os.path.join(ROOT_DIR, "..", "..", "cfd_training_data")
os.makedirs(CFD_DIR, exist_ok=True)

OUT_PATH = os.path.join(CFD_DIR, "cyl_grid.xyz")


def main():
    # Tensor-product grid, ordered x-major then y then z (stable reshape to [Nx,Ny,Nz])
    x_vals = np.linspace(0.0, L_PIPE, Nx)
    y_vals = np.linspace(-R_PIPE, R_PIPE, Ny)
    z_vals = np.linspace(-R_PIPE, R_PIPE, Nz)

    # Build points in the exact loop order expected by reshape:
    # flat_index = ix*(Ny*Nz) + iy*(Nz) + iz
    points = np.empty((Nx * Ny * Nz, 3), dtype=float)
    idx = 0
    for ix, x in enumerate(x_vals):
        for iy, y in enumerate(y_vals):
            for iz, z in enumerate(z_vals):
                points[idx, 0] = x
                points[idx, 1] = y
                points[idx, 2] = z
                idx += 1

    print(f"Generated full Cartesian grid with N = {points.shape[0]} = {Nx} * {Ny} * {Nz}")

    # Write OpenFOAM-friendly "cloud" format: one (x y z) per line in parentheses
    with open(OUT_PATH, "w") as f:
        for x, y, z in points:
            f.write(f"({x:.9e} {y:.9e} {z:.9e})\n")

    print(f"Saved cylindrical grid to: {OUT_PATH}")
    print("NOTE: This includes points outside the pipe radius; use mask.npy for training/visualization.")


if __name__ == "__main__":
    main()
