#!/usr/bin/env python3
"""
Evaluate the trained FNO steady-state model on a single case.

Set RE and U_IN below to values that match a case in your fno_grid/ data.
The script finds the closest matching case, runs inference, reports errors,
and writes to results/:

    comparison_2d.png    – 3-row × 4-col heatmap of the y≈0 centreline slice
                           rows:    Actual | Predicted | |Error|
                           columns: ux     | uy        | uz    | p

    actual_3d.vtk        – full 3-D fluid-domain field (actual)     → ParaView
    predicted_3d.vtk     – full 3-D fluid-domain field (predicted)  → ParaView
"""

import os, sys, csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fno_model.fno_model   import FNO3d
from fno_model.fno_dataset import normalize_re, normalize_uin, pad_to_efficient_grid

# ─────────────────────────────────────────────────────────────────────────────
# USER PARAMETERS — set to values matching a case in cfd_training_data/fno_grid/
# ─────────────────────────────────────────────────────────────────────────────
RE   = 350.0    # Reynolds number
U_IN = 0.55     # Inlet velocity [m/s]

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG (should not need to change)
# ─────────────────────────────────────────────────────────────────────────────
R_PHYS      = 0.5          # pipe radius [m]
CYL_X       = 3.0          # cylinder obstacle centre x [m]
CYL_R       = 0.25         # cylinder obstacle radius  [m]
CFD_ROOT    = os.path.join(ROOT, "..", "cfd_training_data", "fno_grid")
MODEL_PATH  = os.path.join(ROOT, "models_fno", "fno3d_best.pt")
OUT_DIR     = os.path.join(ROOT, "results")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_closest_case(cfd_root, target_re, target_uin):
    """Return (case_id, Re, U_in) of the case whose (Re, U_in) is nearest
    to (target_re, target_uin) in normalised Euclidean space."""
    meta_path = os.path.join(cfd_root, "metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.csv not found in {cfd_root}")

    best, best_dist = None, float("inf")
    with open(meta_path, newline="") as f:
        for row in csv.DictReader(f):
            re  = float(row["Re"])
            uin = float(row["U_in"])
            # Normalise both axes to [0,1] before computing distance
            dr = (re - target_re)    / (1000.0 - 100.0)
            du = (uin - target_uin)  / (1.0    - 0.1)
            d  = (dr**2 + du**2) ** 0.5
            if d < best_dist:
                best_dist = d
                best = (row["case_id"], re, uin)

    return best, best_dist


def load_case(cfd_root, case_id):
    """Load coords (N,3), u (N,3), p (N,), sampled_mask (N,) for one case."""
    d = os.path.join(cfd_root, case_id)
    coords       = np.load(os.path.join(d, "coords.npy"))
    u            = np.load(os.path.join(d, "u.npy"))
    p            = np.load(os.path.join(d, "p.npy"))
    mask_path    = os.path.join(d, "sampled_mask.npy")
    sampled_mask = np.load(mask_path) if os.path.exists(mask_path) \
                   else np.ones(coords.shape[0], dtype=np.float32)
    return coords, u, p, sampled_mask.astype(np.float32)


def build_grid_info(coords, sampled_mask_flat, R_phys):
    """Reproduce the grid info logic from CFDGridFnoDataset._get_grid_info."""
    x_u = np.unique(coords[:, 0])
    y_u = np.unique(coords[:, 1])
    z_u = np.unique(coords[:, 2])
    Nx, Ny, Nz = len(x_u), len(y_u), len(z_u)

    Xg, Yg, Zg = np.meshgrid(x_u, y_u, z_u, indexing="ij")

    if Nx * Ny * Nz == coords.shape[0]:
        idx_map = np.arange(Nx * Ny * Nz).reshape(Nx, Ny, Nz)
    else:
        c2i = {tuple(c): i for i, c in enumerate(coords)}
        idx_map = np.zeros((Nx, Ny, Nz), dtype=int)
        for ix, xv in enumerate(x_u):
            for iy, yv in enumerate(y_u):
                for iz, zv in enumerate(z_u):
                    idx_map[ix, iy, iz] = c2i.get((xv, yv, zv), 0)

    x_norm = (Xg - x_u.min()) / (x_u.max() - x_u.min() + 1e-6)
    y_norm = (Yg - y_u.min()) / (y_u.max() - y_u.min() + 1e-6)
    z_norm = (Zg - z_u.min()) / (z_u.max() - z_u.min() + 1e-6)

    r          = np.sqrt(Yg**2 + Zg**2)
    fluid_mask = (r <= R_phys).astype(np.float32)

    dist_c      = np.sqrt((Xg - CYL_X)**2 + Yg**2 + Zg**2)
    cyl_mask    = (dist_c <= CYL_R).astype(np.float32)

    sm_grid    = sampled_mask_flat[idx_map.reshape(-1)].reshape(Nx, Ny, Nz)
    train_mask = fluid_mask * (1.0 - cyl_mask) * sm_grid

    return (Nx, Ny, Nz), (x_u, y_u, z_u), idx_map, \
           (x_norm.astype(np.float32), y_norm.astype(np.float32),
            z_norm.astype(np.float32)),\
           fluid_mask, cyl_mask, train_mask


def build_input_tensor(Nx, Ny, Nz, idx_map, norms, fluid_mask, cyl_mask, Re, U_in):
    """Build the 7-channel input tensor (unpadded, float32 numpy)."""
    x_norm, y_norm, z_norm = norms
    ones = np.ones_like(fluid_mask)
    return np.stack([
        x_norm, y_norm, z_norm,
        fluid_mask, cyl_mask,
        ones * normalize_re(Re),
        ones * normalize_uin(U_in),
    ], axis=0).astype(np.float32)          # (7, Nx, Ny, Nz)


def run_inference(model, x_np, Nx, Ny, Nz):
    """Pad → forward → crop back to original spatial size.
    Returns (4, Nx, Ny, Nz) float32 numpy array."""
    x_t = torch.from_numpy(x_np).unsqueeze(0)               # (1,7,Nx,Ny,Nz)
    x_t = pad_to_efficient_grid(x_t)                         # (1,7,Np,Ny,Nz)
    with torch.no_grad():
        pred = model(x_t.to(DEVICE)).cpu()                   # (1,4,Np,Ny,Nz)
    return pred[0, :, :Nx, :Ny, :Nz].numpy()                # (4,Nx,Ny,Nz)


# ─────────────────────────────────────────────────────────────────────────────
# VTK writer
# ─────────────────────────────────────────────────────────────────────────────

def write_vtk(path, coords_pts, u_vec, p_scalar):
    """Write a legacy ASCII VTK POLYDATA file."""
    N = coords_pts.shape[0]
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"FNO steady-state field\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {N} float\n")
        for x, y, z in coords_pts:
            f.write(f"{x:.6e} {y:.6e} {z:.6e}\n")
        f.write(f"VERTICES {N} {2*N}\n")
        for i in range(N):
            f.write(f"1 {i}\n")
        f.write(f"POINT_DATA {N}\n")
        f.write("VECTORS velocity float\n")
        for ux, uy, uz in u_vec:
            f.write(f"{ux:.6e} {uy:.6e} {uz:.6e}\n")
        f.write("SCALARS pressure float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for pv in p_scalar:
            f.write(f"{pv:.6e}\n")
    print(f"  Wrote {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2-D slice visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_2d(actual_fields, pred_fields, x_vals, z_vals,
                       y_idx, y_val, train_mask_slice, Re, U_in,
                       out_path):
    """
    actual_fields, pred_fields: each a dict with keys 'ux','uy','uz','p',
        values are (Nx, Nz) arrays at the y≈0 slice, in physical units.
    train_mask_slice: (Nx, Nz) boolean — fluid & sampled region at this slice.
    """
    field_names  = ["ux",  "uy",  "uz",  "p"]
    field_labels = ["$u_x$ [m/s]", "$u_y$ [m/s]", "$u_z$ [m/s]",
                    "$p/\\rho$ [m²/s²]"]
    cmaps        = ["viridis", "RdBu_r", "RdBu_r", "RdBu_r"]

    row_labels = ["Actual", "Predicted", "|Error|"]
    n_rows, n_cols = 3, 4

    # Derive figure size from the physical domain aspect ratio so the pipe
    # cross-section is not stretched or squished.
    x_range = x_vals[-1] - x_vals[0]
    z_range = max(z_vals[-1] - z_vals[0], 1e-6)
    phys_aspect = x_range / z_range          # e.g. 6 for a 6 m pipe of 1 m diameter

    col_w = 5.0                              # subplot column width [inches]
    row_h = max(col_w / phys_aspect, 1.0)   # subplot row height [inches]
    fig_w = n_cols * col_w + 2.0            # +2 for colorbars / margins
    fig_h = n_rows * row_h + 2.5            # +2.5 for suptitle / x-labels

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(fig_w, fig_h),
                             constrained_layout=True)

    fig.suptitle(
        f"FNO Steady-State  |  Re = {Re:.1f},  U_in = {U_in:.3f} m/s"
        f"\nCentreline slice at y ≈ {y_val:.3f} m  (iy = {y_idx})",
        fontsize=13
    )

    extent = [x_vals[0], x_vals[-1], z_vals[0], z_vals[-1]]

    # NaN-out points outside fluid domain for clean display
    nan_mask = ~train_mask_slice.astype(bool)

    for col, (fname, flabel, cmap) in \
            enumerate(zip(field_names, field_labels, cmaps)):

        act  = actual_fields[fname].copy()
        pred = pred_fields[fname].copy()
        err  = np.abs(pred - act)

        act[nan_mask]  = np.nan
        pred[nan_mask] = np.nan
        err[nan_mask]  = np.nan

        # Shared colour limits for actual / predicted rows
        vmin = np.nanmin([act, pred])
        vmax = np.nanmax([act, pred])
        # For diverging fields centre on zero
        if fname in ("uy", "uz", "p"):
            absmax = max(abs(vmin), abs(vmax)) + 1e-12
            vmin, vmax = -absmax, absmax

        for row, (data, row_cmap) in enumerate([
            (act,  cmap),
            (pred, cmap),
            (err,  "hot_r"),
        ]):
            ax = axes[row, col]
            if row < 2:
                im = ax.imshow(data.T, origin="lower", aspect="equal",
                               extent=extent, cmap=row_cmap,
                               vmin=vmin, vmax=vmax)
            else:
                im = ax.imshow(data.T, origin="lower", aspect="equal",
                               extent=extent, cmap=row_cmap,
                               vmin=0, vmax=np.nanmax(err) + 1e-12)

            fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

            # Obstacle circle outline (projection onto x-z plane at y≈0)
            circle = mpatches.Circle(
                (CYL_X, 0.0), CYL_R,
                fill=False, edgecolor="white", linewidth=1.2, linestyle="--"
            )
            ax.add_patch(circle)

            # Pipe wall lines
            ax.axhline( R_PHYS, color="grey", linewidth=0.8, linestyle=":")
            ax.axhline(-R_PHYS, color="grey", linewidth=0.8, linestyle=":")

            ax.set_xlabel("x [m]")
            ax.set_ylabel("z [m]")

            if col == 0:
                ax.set_ylabel(f"{row_labels[row]}\nz [m]")
            if row == 0:
                ax.set_title(flabel, fontsize=11)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Error metrics
# ─────────────────────────────────────────────────────────────────────────────

def report_errors(actual_norm, pred_norm, train_mask):
    """
    actual_norm, pred_norm: (4, Nx, Ny, Nz) in normalised units.
    train_mask: (Nx, Ny, Nz) float32.
    Prints per-field and overall masked MSE and relative L2 error.
    """
    mask = train_mask[np.newaxis]                  # (1, Nx, Ny, Nz)
    n_fluid = mask.sum()
    names = ["ux/U_in", "uy/U_in", "uz/U_in", "p/U_in²"]

    print("\n  ── Error metrics (normalised units, fluid domain only) ──")
    overall_mse = 0.0
    for i, name in enumerate(names):
        diff   = (pred_norm[i] - actual_norm[i]) * train_mask
        sq     = diff ** 2
        mse    = sq.sum() / (n_fluid + 1e-8)
        denom  = ((actual_norm[i] * train_mask) ** 2).sum() / (n_fluid + 1e-8)
        rel_l2 = (mse / (denom + 1e-8)) ** 0.5
        print(f"    {name:>8s}:  MSE = {mse:.4e}   rel-L2 = {rel_l2*100:.2f}%")
        overall_mse += mse

    overall_mse /= 4
    diff_all  = (pred_norm - actual_norm) * mask
    denom_all = (actual_norm * mask) ** 2
    rel_l2_all = (diff_all**2).sum() / (denom_all.sum() + 1e-8)
    print(f"    {'overall':>8s}:  MSE = {overall_mse:.4e}"
          f"   rel-L2 = {rel_l2_all**0.5*100:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Find matching case ─────────────────────────────────────────────
    print(f"\nLooking for Re={RE}, U_in={U_IN} in {CFD_ROOT}...")
    (case_id, case_re, case_uin), dist = find_closest_case(CFD_ROOT, RE, U_IN)
    print(f"  Best match: {case_id}  (Re={case_re:.2f}, U_in={case_uin:.4f})")
    if dist > 0.05:
        print(f"  WARNING: normalised distance = {dist:.4f}.  "
              f"No case closely matches Re={RE}, U_in={U_IN}. "
              f"Consider adjusting the constants.")
    else:
        print(f"  Match quality: good (normalised dist = {dist:.4f})")

    Re   = case_re
    U_in = case_uin

    # ── 2. Load case data ─────────────────────────────────────────────────
    coords, u_flat, p_flat, sm_flat = load_case(CFD_ROOT, case_id)
    (Nx, Ny, Nz), (x_u, y_u, z_u), idx_map, norms, \
        fluid_mask, cyl_mask, train_mask = build_grid_info(coords, sm_flat, R_PHYS)

    print(f"  Grid: ({Nx}, {Ny}, {Nz})  total points: {Nx*Ny*Nz}")

    flat_idx = idx_map.reshape(-1)

    # Actual fields — normalised (same as dataset)
    u_grid = u_flat[flat_idx].reshape(Nx, Ny, Nz, 3) / U_in   # (Nx,Ny,Nz,3)
    p_grid = p_flat[flat_idx].reshape(Nx, Ny, Nz)   / (U_in**2)

    mask_bool = train_mask > 0.5
    p_ref = float(np.mean(p_grid[mask_bool])) if mask_bool.any() else 0.0
    p_grid -= p_ref

    actual_norm = np.stack([
        u_grid[..., 0], u_grid[..., 1], u_grid[..., 2], p_grid
    ], axis=0).astype(np.float32)                              # (4,Nx,Ny,Nz)

    # ── 3. Load model and run inference ───────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            f"Train first with: python3 train.py"
        )

    model = FNO3d(modes_x=16, modes_y=10, modes_z=10,
                  width=32, in_channels=7, out_channels=4, n_layers=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    x_np    = build_input_tensor(Nx, Ny, Nz, idx_map, norms,
                                 fluid_mask, cyl_mask, Re, U_in)
    pred_norm = run_inference(model, x_np, Nx, Ny, Nz)         # (4,Nx,Ny,Nz)

    # ── 4. Error metrics ──────────────────────────────────────────────────
    report_errors(actual_norm, pred_norm, train_mask)

    # ── 5. 2-D heatmap at y≈0 ────────────────────────────────────────────
    iy   = Ny // 2      # index 10, y ≈ 0.026 m
    y_val = float(y_u[iy])
    print(f"\nExtracting centreline slice: iy={iy}, y={y_val:.4f} m")

    # Physical units for display.
    # actual_norm / pred_norm are (4, Nx, Ny, Nz); dim 2 is y → slice at iy.
    actual_2d = {
        "ux": actual_norm[0, :, iy, :] * U_in,     # (Nx, Nz)
        "uy": actual_norm[1, :, iy, :] * U_in,
        "uz": actual_norm[2, :, iy, :] * U_in,
        "p":  actual_norm[3, :, iy, :] * U_in**2,
    }
    pred_2d = {
        "ux": pred_norm[0, :, iy, :] * U_in,
        "uy": pred_norm[1, :, iy, :] * U_in,
        "uz": pred_norm[2, :, iy, :] * U_in,
        "p":  pred_norm[3, :, iy, :] * U_in**2,
    }
    mask_2d = train_mask[:, iy, :]    # (Nx, Nz)

    out_png = os.path.join(OUT_DIR, "comparison_2d.png")
    plot_comparison_2d(actual_2d, pred_2d, x_u, z_u,
                       iy, y_val, mask_2d, Re, U_in, out_png)

    # ── 6. 3-D VTK files ─────────────────────────────────────────────────
    print("\nWriting 3-D VTK files...")

    # Use fluid domain (excluding obstacle interior) for clean ParaView viz
    fluid_pts_mask = (fluid_mask * (1.0 - cyl_mask)) > 0.5    # (Nx,Ny,Nz)
    flat_fluid = fluid_pts_mask.reshape(-1)                    # (N_grid,)

    # Coordinates of fluid points
    Xg, Yg, Zg = np.meshgrid(x_u, y_u, z_u, indexing="ij")
    coords_3d = np.stack([Xg.reshape(-1), Yg.reshape(-1),
                          Zg.reshape(-1)], axis=1)[flat_fluid]  # (N_fluid,3)

    def extract_fluid(field_norm, scale):
        """(4,Nx,Ny,Nz) → (N_fluid,3 or 1) in physical units."""
        return (field_norm * scale).reshape(4, -1).T[flat_fluid]  # (N_fluid,4)

    act_phys  = extract_fluid(actual_norm,
                              np.array([U_in, U_in, U_in, U_in**2])[:, None, None, None])
    pred_phys = extract_fluid(pred_norm,
                              np.array([U_in, U_in, U_in, U_in**2])[:, None, None, None])

    write_vtk(os.path.join(OUT_DIR, "actual_3d.vtk"),
              coords_3d, act_phys[:, :3], act_phys[:, 3])
    write_vtk(os.path.join(OUT_DIR, "predicted_3d.vtk"),
              coords_3d, pred_phys[:, :3], pred_phys[:, 3])

    print(f"\nDone.  All outputs in: {OUT_DIR}/")
    print("  comparison_2d.png   – open with any image viewer")
    print("  actual_3d.vtk       – open in ParaView")
    print("  predicted_3d.vtk    – open in ParaView")
    print("\nParaView tip: open both VTK files, colour by 'velocity' magnitude")
    print("or 'pressure', and use 'Glyph' filter for velocity arrows.")


if __name__ == "__main__":
    main()
