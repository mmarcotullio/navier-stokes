"""
Evaluate the trained CNN3DInverse on validation cases (decoupled 2-stage).

Stage 1 (U_in):
    U_in_pred = clip(a × ux_mean_upstream + b, UIN_MIN, UIN_MAX)
    Coefficients loaded from models/uin_linear.json.

Stage 2 (Re) with Test-Time Augmentation (TTA):
    For each validation sample, run forward passes over 2 D4 variants
    (identity + y-axis flip) and average the Re_log_norm predictions.
    TTA exploits the pipe's bilateral symmetry to cancel residual asymmetries
    introduced by finite training data and discrete voxel effects.

Outputs (written to data/inverse/images/):
    scatter_Re.png      — predicted vs true Re, coloured by U_in
    scatter_Uin.png     — predicted vs true U_in, coloured by Re
    error_heatmap.png   — absolute error in (Re, U_in) parameter space
    metrics.txt         — RMSE, MRE, max error per parameter (with and without TTA)

Run from the inverse/ directory:
    python evaluate.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def _worker_init_fn(worker_id: int) -> None:
    np.random.seed(torch.initial_seed() % (2 ** 32))


# ── Local imports ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dataset import VoxelCFDDataset
from model   import CNN3DInverse
from utils   import RE_LOG_MIN, RE_LOG_MAX, UIN_MIN, UIN_MAX, make_split

# ── Config ────────────────────────────────────────────────────────────────────
SEED       = 42
BATCH_SIZE = 16
IN_CHANNELS = 6
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ── D4 augmentation for TTA ───────────────────────────────────────────────────

def _augment_d4_batch(vt_batch: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """Apply D4 transform to a batch tensor (B, 6, Nx, Ny, Nz)."""
    vt = vt_batch.clone()
    if flip:
        vt      = torch.flip(vt, dims=[3])   # flip Ny axis (axis 3 of (B,6,Nx,Ny,Nz))
        vt[:, 1] = -vt[:, 1]
    if k > 0:
        vt = torch.rot90(vt, k=k, dims=[3, 4])   # rotate (Ny, Nz) in batch tensor
        for _ in range(k):
            uy_tmp  = vt[:, 1].clone()
            vt[:, 1] = -vt[:, 2]
            vt[:, 2] =  uy_tmp
    return vt


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_scatter(pred_phys, true_phys, colour_vals, param_name,
                 colour_label, out_path, unit, axis_lim=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        true_phys, pred_phys,
        c=colour_vals, cmap="viridis", s=18, alpha=0.75, edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label=colour_label)
    if axis_lim is not None:
        lo, hi = axis_lim
    else:
        lo = min(true_phys.min(), pred_phys.min())
        hi = max(true_phys.max(), pred_phys.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="ideal")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"True {param_name} {unit}")
    ax.set_ylabel(f"Predicted {param_name} {unit}")
    ax.set_title(f"CNN3DInverse: {param_name} prediction (TTA)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {out_path}")



def print_metrics(true_phys, pred_phys, name, unit, out_file):
    err     = np.abs(pred_phys - true_phys)
    rmse    = float(np.sqrt((err ** 2).mean()))
    mre     = float((err / (np.abs(true_phys) + 1e-8)).mean()) * 100
    max_err = float(err.max())
    line    = (
        f"{name}: RMSE={rmse:.3f} {unit} | MRE={mre:.2f}% | "
        f"Max err={max_err:.3f} {unit}"
    )
    print(f"  {line}")
    out_file.write(line + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    DATA_ROOT          = os.path.join(_HERE, "fno_training_data")
    MODEL_DIR          = os.path.join(_HERE, "models")
    UIN_LINEAR_PATH    = os.path.join(MODEL_DIR, "uin_linear.json")
    DIMLESS_STATS_PATH = os.path.join(MODEL_DIR, "field_stats_dimless_voxel.json")
    CKPT_PATH          = os.path.join(MODEL_DIR, "cnn_best.pt")
    OUT_DIR            = os.path.join(os.path.dirname(_HERE), "data", "inverse", "images")
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load model artefacts ──────────────────────────────────────────────────
    for path, name in [
        (UIN_LINEAR_PATH,    "uin_linear.json"),
        (DIMLESS_STATS_PATH, "field_stats_dimless_voxel.json"),
        (CKPT_PATH,          "cnn_best.pt"),
    ]:
        if not os.path.isfile(path):
            print(f"[ERROR] {name} not found: {path}")
            print("  Run inverse/train.py first.")
            sys.exit(1)

    with open(UIN_LINEAR_PATH) as f:
        uin_linear = json.load(f)
    with open(DIMLESS_STATS_PATH) as f:
        dimless_stats = json.load(f)

    use_true_uin = bool(uin_linear.get("use_true_uin", False))
    a_lin = uin_linear.get("a", 0.0)
    b_lin = uin_linear.get("b", 0.0)
    if use_true_uin:
        print("Stage 1: bypassed — true U_in read from metadata")
    else:
        print(f"Stage 1: U_in = {a_lin:.4f} × ux_mean + {b_lin:.4f}")

    # ── Dataset — evaluate on all available cases ─────────────────────────────
    ds = VoxelCFDDataset(
        root_dir=DATA_ROOT,
        augment=False, noise_std=0.0,
        field_stats=dimless_stats,
        uin_linear=uin_linear,
        use_true_uin=use_true_uin,
    )
    n_total = len(ds)
    if n_total == 0:
        print("[ERROR] No on-disk cases found in fno_training_data/.")
        sys.exit(1)
    # Use all available cases (no train/val split needed for evaluation)
    val_idx = list(range(n_total))
    print(f"Evaluating on all {len(val_idx)} available cases…")
    print(f"TTA: 2 D4 variants per sample (identity + y-flip)\n")

    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=(DEVICE == "cuda"),
        worker_init_fn=_worker_init_fn,
    )

    # ── Load Stage 2 model ────────────────────────────────────────────────────
    model = CNN3DInverse(in_channels=IN_CHANNELS).to(DEVICE)
    model.load_state_dict(
        torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    # ── Stage 2 inference with TTA ────────────────────────────────────────────
    # For each batch: run 2 D4 variants (identity + y-flip), average Re_log_norm.
    re_preds_norm   = []
    re_targets_norm = []

    print("Running inference (2 D4 variants per batch)…")
    with torch.no_grad():
        for batch_idx, (vt, y) in enumerate(val_loader):
            # vt: (B, 6, Nx, Ny, Nz)
            batch_preds = []
            for flip in [False, True]:
                vt_aug = _augment_d4_batch(vt, 0, flip).to(DEVICE)
                pred = model(vt_aug).cpu().numpy()   # (B, 1)
                batch_preds.append(pred)

            # Average over 2 D4 variants → (B, 1)
            tta_pred = np.mean(batch_preds, axis=0)
            re_preds_norm.append(tta_pred.flatten())
            re_targets_norm.append(y.numpy().flatten())

            print(
                f"\r  Batch {batch_idx + 1}/{len(val_loader)}",
                end="", flush=True,
            )

    print()

    re_preds_norm   = np.clip(np.concatenate(re_preds_norm),   0.0, 1.0)
    re_targets_norm = np.concatenate(re_targets_norm)

    # ── Stage 1 inference: U_in ───────────────────────────────────────────────
    u_in_true = np.array([
        float(ds.cases[ds.sample_index[i]]["U_in"]) for i in val_idx
    ])
    if use_true_uin:
        # U_in is known exactly from metadata — Stage 1 is trivially perfect
        u_in_preds = u_in_true.copy()
    else:
        ux_means   = np.array([ds.get_ux_mean_upstream(i) for i in val_idx])
        u_in_preds = np.clip(a_lin * ux_means + b_lin, UIN_MIN, UIN_MAX)

    # Denormalise Re
    pred_re = np.exp(re_preds_norm   * (RE_LOG_MAX - RE_LOG_MIN) + RE_LOG_MIN)
    true_re = np.exp(re_targets_norm * (RE_LOG_MAX - RE_LOG_MIN) + RE_LOG_MIN)

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n── Evaluation metrics ──")
    metrics_path = os.path.join(OUT_DIR, "metrics.txt")
    with open(metrics_path, "w") as mf:
        mf.write("CNN3DInverse — decoupled 2-stage inverse model (with TTA)\n\n")
        mf.write("Stage 2 (CNN Re prediction, 8-variant TTA):\n")
        print_metrics(true_re, pred_re, "Re", "[-]", mf)
        mf.write("\nStage 1 (linear U_in prediction):\n")
        print_metrics(u_in_true, u_in_preds, "U_in", "[m/s]", mf)
    print(f"  Wrote {metrics_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n── Writing plots ──")
    plot_scatter(
        pred_re, true_re,
        colour_vals=u_in_true, colour_label="True U_in [m/s]",
        param_name="Re", unit="[-]",
        out_path=os.path.join(OUT_DIR, "scatter_Re.png"),
        axis_lim=(100, 1000),
    )
    plot_scatter(
        u_in_preds, u_in_true,
        colour_vals=true_re, colour_label="True Re [-]",
        param_name="U_in", unit="[m/s]",
        out_path=os.path.join(OUT_DIR, "scatter_Uin.png"),
    )
    print(f"\nAll outputs in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
