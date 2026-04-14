"""
Training script for CNN3DInverse surrogate — FNO-synthetic data, true U_in.

Data source
-----------
FNO-generated synthetic data in inverse/fno_training_data/ (created by
generate_fno_data.py).  3,000 cases: 1,000 Re values × 3 U_in values.

Stage 1 (U_in)
--------------
Bypassed.  U_in is read directly from metadata.csv (use_true_uin=True).
This is exact — no regression error.  Dimensionless channels are exact.
A trivial uin_linear.json {"a":0,"b":0,"use_true_uin":true} is written so
that evaluate.py and downstream code can detect the mode automatically.

Stage 2 (CNN3DInverse — Re prediction)
---------------------------------------
Construct 6 dimensionless input channels using true U_in:
    ux/U_in, uy/U_in, uz/U_in, p/U_in², dp_dx/U_in², |ω|/U_in
Feed (6, Nx, Ny, Nz) voxel tensor into CNN3DInverse to predict Re_log_norm.

Loss: Huber(delta=0.05) + barrier.
Augmentation: D4 dihedral group (4 rotations × 2 reflections = 8 variants).
BN calibration: 20 train-mode forward passes before each val loop to
stabilise BatchNorm running statistics and prevent val-loss oscillation.

Workflow
--------
1. Scan valid cases; case-level 90/10 train/val split (seed=42).
2. Compute 6-channel dimensionless field stats from training cases (true U_in).
3. Build datasets with augment=True (train), augment=False (val).
4. Train CNN3DInverse with AdamW + warmup/cosine.
5. BN calibration pass → validate → early stopping (patience=120).
6. Save best checkpoint.

Run from the inverse/ directory:
    python generate_fno_data.py   # once
    python train.py
"""

import json
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# ── Local imports ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dataset import VoxelCFDDataset
from model   import CNN3DInverse
from utils   import (
    RE_LOG_MIN, RE_LOG_MAX, UIN_MIN, UIN_MAX,
    make_split, re_loss, rmse_re_physical,
)

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_EPOCHS            = 1200
EARLY_STOP_PATIENCE = 120
BATCH_SIZE          = 32
LR                  = 3e-4   # Lower than PointNet++ (deeper BN chain)
WEIGHT_DECAY        = 1e-3   # 10× stronger; ~7.5M params / 900 cases
WARMUP_EPOCHS       = 40
COSINE_T_MAX        = 600
NOISE_STD           = 0.015  # Gaussian noise on all 6 channels after z-score
IN_CHANNELS         = 6
SEED                = 42


# ── Worker seeding ────────────────────────────────────────────────────────────

def _worker_init_fn(worker_id: int) -> None:
    """Seed numpy independently per DataLoader worker."""
    np.random.seed(torch.initial_seed() % (2 ** 32))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("--- CNN3DInverse Training (Decoupled 2-Stage) ---")
    print(f"Device: {device}")

    # ── Paths ─────────────────────────────────────────────────────────────────
    DATA_ROOT          = os.path.join(_HERE, "fno_training_data")
    MODEL_DIR          = os.path.join(_HERE, "models")
    UIN_LINEAR_PATH    = os.path.join(MODEL_DIR, "uin_linear.json")
    DIMLESS_STATS_PATH = os.path.join(MODEL_DIR, "field_stats_dimless_voxel.json")
    SAVE_PATH          = os.path.join(MODEL_DIR, "cnn_best.pt")
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"\nData root : {DATA_ROOT}")

    # Dummy uin_linear used for initial scan (values ignored when use_true_uin=True)
    _DUMMY_UIN = {"a": 0.0, "b": 0.0}

    # ── Step 1: scan valid cases ───────────────────────────────────────────────
    try:
        ds_raw = VoxelCFDDataset(
            root_dir=DATA_ROOT,
            augment=False,
            noise_std=0.0,
            field_stats=None,
            uin_linear=_DUMMY_UIN,
            use_true_uin=True,
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("  Run: python generate_fno_data.py")
        sys.exit(1)

    n_total = len(ds_raw)
    print(f"Total valid cases: {n_total}")
    if n_total == 0:
        print("[STOPPING] Dataset is empty.")
        sys.exit(1)

    # ── Step 2: 90/10 case-level train/val split ───────────────────────────────
    train_idx, val_idx = make_split(n_total)
    print(f"Train: {len(train_idx)}  |  Val: {len(val_idx)}")

    # ── Step 3: Stage 1 — bypassed (true U_in from metadata) ─────────────────
    # Data is FNO-generated with known exact U_in values.  No regression needed.
    print(f"\n[Stage 1] Bypassed — using true U_in from metadata (use_true_uin=True)")
    uin_linear = {"a": 0.0, "b": 0.0, "use_true_uin": True}
    with open(UIN_LINEAR_PATH, "w") as f:
        json.dump(uin_linear, f, indent=2)
    print(f"  Saved → {UIN_LINEAR_PATH}")

    # ── Step 4: dimensionless field statistics (6 channels, training only) ────
    train_case_cis = [ds_raw.sample_index[i] for i in train_idx]
    print(f"\n[Stage 2] Computing 6-channel dimensionless field stats from "
          f"{len(train_case_cis)} training cases…")
    dimless_stats = VoxelCFDDataset.compute_dimless_stats(
        root_dir=DATA_ROOT,
        metadata_csv="metadata.csv",
        train_case_cis=train_case_cis,
        uin_linear=uin_linear,
        use_true_uin=True,
    )
    with open(DIMLESS_STATS_PATH, "w") as f:
        json.dump(dimless_stats, f, indent=2)
    print(f"  Saved → {DIMLESS_STATS_PATH}")
    print(f"  mean: {[f'{v:.4f}' for v in dimless_stats['mean']]}")
    print(f"  std : {[f'{v:.4f}' for v in dimless_stats['std']]}")

    # ── Step 5: rebuild datasets ───────────────────────────────────────────────
    ds_train = VoxelCFDDataset(
        root_dir=DATA_ROOT,
        augment=True, noise_std=NOISE_STD,
        field_stats=dimless_stats,
        uin_linear=uin_linear,
        use_true_uin=True,
    )
    ds_val = VoxelCFDDataset(
        root_dir=DATA_ROOT,
        augment=False, noise_std=0.0,
        field_stats=dimless_stats,
        uin_linear=uin_linear,
        use_true_uin=True,
    )

    train_loader = DataLoader(
        Subset(ds_train, train_idx),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=(device == "cuda"),
        worker_init_fn=_worker_init_fn,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        Subset(ds_val, val_idx),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=(device == "cuda"),
        worker_init_fn=_worker_init_fn,
        persistent_workers=True,
    )

    # ── Step 6: model ──────────────────────────────────────────────────────────
    model = CNN3DInverse(in_channels=IN_CHANNELS).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: CNN3DInverse(in_channels={IN_CHANNELS})  —  {n_params:,} params")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(COSINE_T_MAX - WARMUP_EPOCHS, 1),
        eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS],
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss     = float("inf")
    epochs_no_improve = 0

    print(f"\n--- Starting Training ---")
    print(f"Max epochs: {N_EPOCHS}  |  Patience: {EARLY_STOP_PATIENCE}")
    print(f"Batch: {BATCH_SIZE}  |  LR: {LR}  |  WD: {WEIGHT_DECAY}  |  Noise: {NOISE_STD}\n")

    n_train_batches = len(train_loader)
    n_val_batches   = len(val_loader)

    for epoch in range(N_EPOCHS):

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss, n_batches = 0.0, 0

        for vt, y in train_loader:
            vt, y = vt.to(device), y.to(device)   # (B, 6, Nx, Ny, Nz), (B, 1)

            optimizer.zero_grad()
            pred = model(vt)                        # (B, 1)
            loss = re_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches  += 1

            print(
                f"\rEpoch {epoch:4d}  train {n_batches:3d}/{n_train_batches}"
                f"  loss {train_loss / n_batches:.4f}",
                end="", flush=True,
            )

        if n_batches:
            train_loss /= n_batches

        if n_batches and not math.isfinite(train_loss):
            print(f"\n[ERROR] Non-finite train loss at epoch {epoch}. Stopping.")
            break

        # ── BN calibration (stabilise running stats before val) ───────────────
        # Run N_BN_CAL train-mode forward passes (no grad) so BatchNorm running
        # mean/var reflect the current weight state before switching to eval.
        N_BN_CAL = min(20, len(train_loader))
        model.train()
        with torch.no_grad():
            for _cal_i, (vt_cal, _) in enumerate(train_loader):
                if _cal_i >= N_BN_CAL:
                    break
                model(vt_cal.to(device))

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for i, (vt, y) in enumerate(val_loader):
                vt, y = vt.to(device), y.to(device)
                pred  = model(vt)                    # (B, 1)
                val_loss += re_loss(pred, y).item()
                val_preds.append(pred.cpu().numpy())
                val_targets.append(y.cpu().numpy())

                print(
                    f"\rEpoch {epoch:4d}  val   {i + 1:3d}/{n_val_batches}",
                    end="", flush=True,
                )

        if val_preds:
            val_loss /= len(val_preds)

        scheduler.step()

        # Physical-unit Re RMSE for monitoring
        vp = np.concatenate(val_preds,   axis=0).flatten() if val_preds   else np.zeros(1)
        vt_arr = np.concatenate(val_targets, axis=0).flatten() if val_targets else np.zeros(1)
        rmse_re = rmse_re_physical(vp, vt_arr)

        print(
            f"\rEpoch {epoch:4d}: Train {train_loss:.6f} | Val {val_loss:.6f}"
            f" | Re RMSE {rmse_re:6.1f}"
            f" | LR {optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), SAVE_PATH)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs).")
            break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoint    → {SAVE_PATH}")
    print(f"U_in linear   → {UIN_LINEAR_PATH}")
    print(f"Dimless stats → {DIMLESS_STATS_PATH}")


if __name__ == "__main__":
    main()
