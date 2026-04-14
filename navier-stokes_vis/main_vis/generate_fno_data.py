"""
Self-contained geometry helpers for the navier-stokes_vis backend.

Provides build_geometry() and fno_to_voxel() for use by server.py.
No external project dependencies — all constants and helpers are inlined.
"""

import numpy as np

# ── Physics constants (must match training setup) ─────────────────────────────
PIPE_L = 10.0   # pipe length [m]
PIPE_R = 0.5    # pipe radius [m]
CYL_X  = 3.0   # sphere centre x [m]
CYL_R  = 0.25  # sphere radius [m]

NX, NY, NZ = 265, 20, 20   # FNO native training resolution


# ── Finite-difference helper ──────────────────────────────────────────────────

def _central_diff(arr: np.ndarray, ax: int, h: float) -> np.ndarray:
    """Second-order finite differences along axis ax with uniform spacing h."""
    out = np.empty_like(arr)
    n = arr.shape[ax]

    def _sl(axis, idx_or_slice):
        s = [slice(None)] * 3
        s[axis] = idx_or_slice
        return tuple(s)

    if n >= 3:
        out[_sl(ax, slice(1, -1))] = (
            arr[_sl(ax, slice(2, None))] - arr[_sl(ax, slice(None, -2))]
        ) / (2.0 * h)
        out[_sl(ax, 0)] = (
            -3.0 * arr[_sl(ax, 0)]
            + 4.0 * arr[_sl(ax, 1)]
            -       arr[_sl(ax, 2)]
        ) / (2.0 * h)
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


# ── Geometry builder ──────────────────────────────────────────────────────────

def build_geometry():
    """Build fixed geometry arrays for the 265×20×20 Cartesian grid.

    Returns (geom_np, full_fluid, dx, dy, dz).
    Coordinate convention matches fno_dataset.py: each axis normalised to [0, 1].
    """
    xs = np.linspace(0.0,    PIPE_L, NX, dtype=np.float32)
    ys = np.linspace(-PIPE_R, PIPE_R, NY, dtype=np.float32)
    zs = np.linspace(-PIPE_R, PIPE_R, NZ, dtype=np.float32)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    x_norm = (X - xs.min()) / (xs.max() - xs.min())
    y_norm = (Y - ys.min()) / (ys.max() - ys.min())
    z_norm = (Z - zs.min()) / (zs.max() - zs.min())

    R2D         = np.sqrt(Y ** 2 + Z ** 2)
    dist_sphere = np.sqrt((X - CYL_X) ** 2 + Y ** 2 + Z ** 2)
    fluid_mask  = (R2D <= PIPE_R).astype(np.float32)
    cyl_mask    = (dist_sphere <= CYL_R).astype(np.float32)

    full_fluid = (R2D <= PIPE_R) & (dist_sphere > CYL_R)

    geom_np = np.stack([x_norm, y_norm, z_norm, fluid_mask, cyl_mask])

    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    dz = float(zs[1] - zs[0])

    return geom_np, full_fluid, dx, dy, dz


# ── Voxel converter ───────────────────────────────────────────────────────────

def fno_to_voxel(
    fno_out_np: np.ndarray,
    U_in: float,
    full_fluid: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """Convert FNO output (4, Nx, Ny, Nz) to a raw 6-channel voxel.

    FNO output channels: [ux/U_in, uy/U_in, uz/U_in, p/U_in²]
    Voxel channels:      [ux, uy, uz, p_centered, dp_dx, |ω|]
    """
    ux  = fno_out_np[0] * U_in
    uy  = fno_out_np[1] * U_in
    uz  = fno_out_np[2] * U_in
    p_c = fno_out_np[3] * (U_in ** 2)

    dp_dx = _central_diff(p_c, ax=0, h=dx)

    omega_x  = _central_diff(uz, 1, dy) - _central_diff(uy, 2, dz)
    omega_y  = _central_diff(ux, 2, dz) - _central_diff(uz, 0, dx)
    omega_z  = _central_diff(uy, 0, dx) - _central_diff(ux, 1, dy)
    vort_mag = np.sqrt(omega_x ** 2 + omega_y ** 2 + omega_z ** 2)

    voxel = np.stack([ux, uy, uz, p_c, dp_dx, vort_mag]).astype(np.float32)
    voxel[:, ~full_fluid] = 0.0

    return voxel
