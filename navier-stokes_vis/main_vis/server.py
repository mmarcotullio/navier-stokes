#!/usr/bin/env python3
"""
Flask backend for FNO 3D CFD point-cloud visualiser.
Loads the trained FNO3d model and serves inference results as JSON.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from forward_tool import FNO3d, normalize_re, normalize_uin, pad_to_efficient_grid
import inverse_tool as _inv_tool

# ── Import training-resolution geometry helpers ────────────────────────────────
from generate_fno_data import build_geometry, fno_to_voxel

# ── Constants (matching training setup) ──────────────────────────────────────
L_PIPE = 10.0   # pipe length [m]
R_PIPE = 0.5    # pipe radius [m]
CYL_X  = 3.0   # cylinder centre x [m]
CYL_R  = 0.25  # cylinder radius [m]

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_VIS_DIR   = os.path.dirname(THIS_DIR)
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(_VIS_DIR, "models", "fno3d_best_forward.pt")

# ── Load forward model at startup ─────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH} onto {DEVICE}...")
_model = FNO3d(modes_x=16, modes_y=10, modes_z=10,
               width=32, in_channels=7, out_channels=4, n_layers=4)
_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
_model.to(DEVICE).eval()
print("Model loaded.")

# ── Load inverse model at startup ─────────────────────────────────────────────
_inv_tool.load_model(DEVICE)
print("Inverse model loaded.")

# ── Cache training-resolution geometry (265×20×20) ───────────────────────────
_inv_geom_np, _inv_full_fluid, _inv_dx, _inv_dy, _inv_dz = build_geometry()
_INV_NX = _inv_geom_np.shape[1]  # 265
_INV_NY = _inv_geom_np.shape[2]  # 20
_INV_NZ = _inv_geom_np.shape[3]  # 20
print(f"Inverse geometry cached: {_INV_NX}×{_INV_NY}×{_INV_NZ}")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


def _run_forward(Re: float, U_in: float, Ny: int, Nz: int) -> dict:
    """Build input grid, run FNO inference, return raw field arrays."""
    # Derive Nx so spacing is isotropic in cross-section
    dy = (2.0 * R_PIPE) / (Ny - 1)
    Nx = int(L_PIPE / dy) + 1

    x_u = np.linspace(0.0, L_PIPE, Nx)
    y_u = np.linspace(-R_PIPE, R_PIPE, Ny)
    z_u = np.linspace(-R_PIPE, R_PIPE, Nz)

    Xg, Yg, Zg = np.meshgrid(x_u, y_u, z_u, indexing="ij")  # (Nx, Ny, Nz)

    # Normalised coordinates [0, 1]
    x_norm = (Xg - x_u.min()) / (x_u.max() - x_u.min() + 1e-6)
    y_norm = (Yg - y_u.min()) / (y_u.max() - y_u.min() + 1e-6)
    z_norm = (Zg - z_u.min()) / (z_u.max() - z_u.min() + 1e-6)

    # Masks
    r          = np.sqrt(Yg**2 + Zg**2)
    fluid_mask = (r <= R_PIPE).astype(np.float32)
    dist_c     = np.sqrt((Xg - CYL_X)**2 + Yg**2 + Zg**2)
    cyl_mask   = (dist_c <= CYL_R).astype(np.float32)

    ones = np.ones_like(fluid_mask)
    x_np = np.stack([
        x_norm, y_norm, z_norm,
        fluid_mask, cyl_mask,
        ones * normalize_re(Re),
        ones * normalize_uin(U_in),
    ], axis=0).astype(np.float32)  # (7, Nx, Ny, Nz)

    # Inference
    x_t = torch.from_numpy(x_np).unsqueeze(0)   # (1, 7, Nx, Ny, Nz)
    x_t = pad_to_efficient_grid(x_t)
    with torch.no_grad():
        pred = _model(x_t.to(DEVICE)).cpu()
    pred = pred[0, :, :Nx, :Ny, :Nz].numpy()    # (4, Nx, Ny, Nz)

    # Physical units
    ux = pred[0] * U_in
    uy = pred[1] * U_in
    uz = pred[2] * U_in
    p  = pred[3] * (U_in ** 2)

    fluid_only = (fluid_mask > 0.5) & (cyl_mask < 0.5)

    return {
        "ux": ux, "uy": uy, "uz": uz, "p": p,
        "x_u": x_u, "y_u": y_u, "z_u": z_u,
        "Xg": Xg, "Yg": Yg, "Zg": Zg,
        "fluid_mask": fluid_mask, "cyl_mask": cyl_mask,
        "fluid_only": fluid_only,
        "Nx": Nx, "Ny": Ny, "Nz": Nz,
    }


def build_and_infer(Re: float, U_in: float, Ny: int, Nz: int) -> dict:
    """Build input grid, run FNO inference, return fluid-domain point data."""
    raw = _run_forward(Re, U_in, Ny, Nz)
    ux, uy, uz, p = raw["ux"], raw["uy"], raw["uz"], raw["p"]
    Xg, Yg, Zg   = raw["Xg"], raw["Yg"], raw["Zg"]
    fluid_mask    = raw["fluid_mask"]
    cyl_mask      = raw["cyl_mask"]
    fluid_only    = raw["fluid_only"]
    Nx, Ny, Nz    = raw["Nx"], raw["Ny"], raw["Nz"]
    x_u, y_u, z_u = raw["x_u"], raw["y_u"], raw["z_u"]

    # ── y ≈ 0 slice for the 2-D heatmap (always full, no downsampling) ───────
    iy = Ny // 2
    y_val = float(y_u[iy])
    sl_fluid = (fluid_mask[:, iy, :] > 0.5) & (cyl_mask[:, iy, :] < 0.5)
    sl_sel = sl_fluid.reshape(-1)
    sl_vmag = np.sqrt(ux[:, iy, :]**2 + uy[:, iy, :]**2 + uz[:, iy, :]**2)
    slice_data = {
        "x":    Xg[:, iy, :].reshape(-1)[sl_sel].astype(np.float32).tolist(),
        "z":    Zg[:, iy, :].reshape(-1)[sl_sel].astype(np.float32).tolist(),
        "vmag": sl_vmag.reshape(-1)[sl_sel].astype(np.float32).tolist(),
        "p":    p[:, iy, :].reshape(-1)[sl_sel].astype(np.float32).tolist(),
        "y_val": y_val,
        "Nx": Nx, "Nz": Nz,
    }

    # Keep only fluid domain (inside pipe, outside cylinder)
    sel = fluid_only.reshape(-1)

    X    = Xg.reshape(-1)[sel]
    Y    = Yg.reshape(-1)[sel]
    Z    = Zg.reshape(-1)[sel]
    UX   = ux.reshape(-1)[sel]
    UY   = uy.reshape(-1)[sel]
    UZ   = uz.reshape(-1)[sel]
    P    = p.reshape(-1)[sel]
    vmag = np.sqrt(UX**2 + UY**2 + UZ**2)

    return {
        "x":    X.astype(np.float32).tolist(),
        "y":    Y.astype(np.float32).tolist(),
        "z":    Z.astype(np.float32).tolist(),
        "ux":   UX.astype(np.float32).tolist(),
        "uy":   UY.astype(np.float32).tolist(),
        "uz":   UZ.astype(np.float32).tolist(),
        "p":    P.astype(np.float32).tolist(),
        "vmag": vmag.astype(np.float32).tolist(),
        "slice": slice_data,
        "meta": {
            "Nx": Nx, "Ny": Ny, "Nz": Nz,
            "Re": Re, "U_in": U_in,
            "n_points": len(X),
        },
    }


@app.route("/api/predict")
def predict():
    try:
        Re   = float(request.args.get("Re",   500))
        U_in = float(request.args.get("U_in", 0.5))
        Ny   = int(  request.args.get("Ny",   20))
        Nz   = int(  request.args.get("Nz",   20))

        Re   = float(np.clip(Re,   100,  1000))
        U_in = float(np.clip(U_in, 0.1,  1.0))
        Ny   = max(9, int(Ny))
        Nz   = max(9, int(Nz))
        # Snap to odd so the grid always includes y=0 and z=0
        if Ny % 2 == 0: Ny += 1
        if Nz % 2 == 0: Nz += 1

        data = build_and_infer(Re, U_in, Ny, Nz)
        return jsonify(data)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/inverse")
def inverse():
    try:
        Re   = float(np.clip(float(request.args.get("Re",   500)), 100, 1000))
        U_in = float(np.clip(float(request.args.get("U_in", 0.5)), 0.1,  1.0))

        # Build FNO input at training resolution (265×20×20)
        ones = np.ones((_INV_NX, _INV_NY, _INV_NZ), dtype=np.float32)
        inp  = np.concatenate([
            _inv_geom_np,                           # (5,265,20,20) — x/y/z/fluid/cyl
            ones[None] * normalize_re(Re),           # (1,265,20,20) — Re_norm
            ones[None] * normalize_uin(U_in),        # (1,265,20,20) — U_in_norm
        ], axis=0)[None]                             # (1,7,265,20,20)

        inp_t = pad_to_efficient_grid(torch.from_numpy(inp))
        with torch.no_grad():
            out_t = _model(inp_t.to(DEVICE)).cpu()
        fno_out = out_t[0, :, :_INV_NX, :_INV_NY, :_INV_NZ].numpy()  # (4,265,20,20)

        voxel_raw = fno_to_voxel(fno_out, U_in, _inv_full_fluid, _inv_dx, _inv_dy, _inv_dz)
        re_pred   = _inv_tool.predict_from_voxel_raw(voxel_raw, U_in, DEVICE)

        nu_pred = U_in / re_pred   # ν = U_in*D/Re, D=1m
        nu_true = U_in / Re
        return jsonify({
            "re_pred": round(re_pred, 2),
            "re_true": round(Re, 2),
            "nu_pred": nu_pred,
            "nu_true": nu_true,
            "re_mae":  round(abs(re_pred - Re), 2),
            "nu_mae":  abs(nu_pred - nu_true),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "device": DEVICE})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
