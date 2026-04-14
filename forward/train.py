import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

try:
    from fno_model.fno_model import FNO3d
    from fno_model.fno_dataset import CFDGridFnoDataset
except ImportError:
    print("Error: Could not import FNO3d or CFDGridFnoDataset.")
    print("Make sure you are running this from the folder containing 'fno_model/'")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
N_EPOCHS = 2000          # upper safety cap — early stopping will end it sooner
EARLY_STOP_PATIENCE = 60 # stop if val loss doesn't improve for this many epochs
BATCH_SIZE = 8
WARMUP_EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4


def masked_relative_l2(pred, target, mask):
    """
    Relative L2 loss restricted to the active fluid domain.
    Matches Li et al. (2020) LpLoss with p=2, relative=True.

    loss = mean_over_batch( ||pred_masked - target_masked||_F
                            / ||target_masked||_F )

    pred, target : (B, C, X, Y, Z)
    mask         : (B, X, Y, Z)  — 1 inside fluid, 0 elsewhere
    """
    if mask.dim() == 4:
        mask = mask.unsqueeze(1)  # (B, 1, X, Y, Z)

    B = pred.shape[0]
    pred_m   = (pred   * mask).view(B, -1)
    target_m = (target * mask).view(B, -1)

    diff_norm = (pred_m - target_m).norm(p=2, dim=1)   # (B,)
    tgt_norm  = target_m.norm(p=2, dim=1)              # (B,)

    return (diff_norm / (tgt_norm + 1e-8)).mean()


def main():
    # Reproducibility
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- FNO Steady-State Training ---")
    print(f"Using device: {device}")

    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)   # project root (one level above forward/)
    CFD_ROOT = os.path.join(ROOT_DIR, "cfd_training_data", "fno_grid")
    MODEL_DIR = os.path.join(BASE_DIR, "models_fno")
    os.makedirs(MODEL_DIR, exist_ok=True)

    possible_csv_paths = [
        os.path.join(CFD_ROOT, "metadata.csv"),
        os.path.join(ROOT_DIR, "cfd_training_data", "numpy", "metadata.csv"),
        os.path.join(BASE_DIR, "metadata.csv"),
    ]

    METADATA_PATH = None
    for p in possible_csv_paths:
        if os.path.exists(p):
            METADATA_PATH = p
            break

    if METADATA_PATH is None:
        print(f"\n[CRITICAL ERROR] Could not find 'metadata.csv'.")
        print(f"Checked: {possible_csv_paths}")
        return

    print(f"Data Root : {CFD_ROOT}")
    print(f"Metadata  : {METADATA_PATH}")

    # Dataset — one sample per case (the converged equilibrium)
    try:
        dataset = CFDGridFnoDataset(
            root_dir=CFD_ROOT,
            metadata_csv=METADATA_PATH,
            pad_data=True,
            augment=True,
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to initialise dataset: {e}")
        return

    n_total = len(dataset)
    print(f"Total cases (samples): {n_total}")

    if n_total == 0:
        print("\n[STOPPING] Dataset is empty.")
        print("Check that fno_grid/ contains case_XXX/ folders with u.npy and p.npy.")
        return

    # Case-level 90/10 train/val split
    case_ids = list(dataset.sample_index)  # one entry per case
    n_cases = len(case_ids)
    shuffled = np.random.permutation(n_cases)
    n_train = max(1, int(0.9 * n_cases))
    if n_train == n_cases and n_cases > 1:
        n_train -= 1

    train_idx = shuffled[:n_train].tolist()
    val_idx = shuffled[n_train:].tolist()

    print(f"Train cases: {len(train_idx)}  |  Val cases: {len(val_idx)}")

    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_idx),
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(val_idx),
        num_workers=2, pin_memory=True,
    )

    # Model — 7 input channels: x, y, z, fluid_mask, cyl_mask, Re_norm, U_in_norm
    model = FNO3d(
        modes_x=16, modes_y=10, modes_z=10,
        width=32,
        in_channels=7,
        out_channels=4,
        n_layers=4,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # T_max should reflect when the model actually converges, not the hard cap.
    # Previous runs converged ~300 epochs, so 400 gives the cosine decay time
    # to reach eta_min before early stopping fires.
    COSINE_T_MAX = 400
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(COSINE_T_MAX - WARMUP_EPOCHS, 1), eta_min=1e-6
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-2, total_iters=WARMUP_EPOCHS
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS]
    )

    best_loss = float('inf')
    epochs_no_improve = 0
    save_path = os.path.join(MODEL_DIR, "fno3d_best.pt")

    print(f"\n--- Starting Training ---")
    print(f"Max epochs: {N_EPOCHS}  |  Early stop patience: {EARLY_STOP_PATIENCE}")
    print(f"Batch: {BATCH_SIZE}  |  LR: {LR}")
    print(f"Model saved to: {save_path}\n")

    for epoch in range(N_EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0
        batches = 0

        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = masked_relative_l2(pred, y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            batches += 1

        if batches > 0:
            train_loss /= batches

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                pred = model(x)
                loss = masked_relative_l2(pred, y, mask)
                val_loss += loss.item()
                val_batches += 1

        if val_batches > 0:
            val_loss /= val_batches

        scheduler.step()

        print(f"Epoch {epoch:4d}: Train {train_loss:.6f} | Val {val_loss:.6f} | "
              f"LR {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs).")
            break

    print(f"\nTraining complete. Best val relative-L2: {best_loss:.6f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
