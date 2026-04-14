"""
Shared utilities for the CNN inverse surrogate.

Contents
--------
- Physics constants and label normalisation (Re log-scale)
- make_split : reproducible 90/10 train/val split
- re_loss    : Huber(delta=0.05) + barrier loss for Re prediction
- rmse_re_physical : physical-unit Re RMSE from normalised predictions
- fps_numpy  : included for compatibility (unused by CNN model)
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

# ── Physics constants ──────────────────────────────────────────────────────────
RE_MIN,  RE_MAX  = 100.0, 1000.0
UIN_MIN, UIN_MAX = 0.1,   1.0

RE_LOG_MIN = math.log(RE_MIN)   # log(100)  ≈ 4.60517
RE_LOG_MAX = math.log(RE_MAX)   # log(1000) ≈ 6.90776

# CFD geometry constants (must stay in sync with dataset)
CYL_X  = 3.0   # obstacle sphere centre, x [m]
CYL_R  = 0.25  # obstacle sphere radius    [m]
PIPE_R = 0.5   # pipe radius               [m]
PIPE_L = 10.0  # pipe length               [m]  (x from 0 → PIPE_L)

BARRIER_WEIGHT = 0.1   # weight of out-of-range barrier in re_loss


# ── Label normalisation ────────────────────────────────────────────────────────

def normalize_re(re: float) -> float:
    """
    Log-scale normalisation: Re=100→0, Re≈316→0.5, Re=1000→1.

    Log-scale equalises the loss gradient across relative Re changes so that
    distinguishing Re=100 from Re=200 is penalised as strongly as Re=500 from
    Re=1000. Linear normalisation starves the low-Re regime where the signal
    is already weakest (small wake).
    """
    return (math.log(re) - RE_LOG_MIN) / (RE_LOG_MAX - RE_LOG_MIN)


def denorm_re(re_norm: "np.ndarray | float") -> "np.ndarray | float":
    """Inverse of normalize_re (log-scale)."""
    return np.exp(np.asarray(re_norm) * (RE_LOG_MAX - RE_LOG_MIN) + RE_LOG_MIN)


# ── Train/val split ───────────────────────────────────────────────────────────

def make_split(
    n_total: int,
    seed: int = 42,
    val_frac: float = 0.1,
) -> "tuple[list, list]":
    """
    Reproducible 90/10 train/val split.

    Uses np.random.default_rng(seed) internally so the result is independent
    of global random state and identical across call sites given the same args.

    Args:
        n_total  : total number of samples.
        seed     : RNG seed (default 42).
        val_frac : fraction of samples for validation (default 0.1).

    Returns:
        (train_idx, val_idx) — lists of integer indices into [0, n_total).
    """
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_total)
    n_train = max(1, int((1.0 - val_frac) * n_total))
    if n_train == n_total and n_total > 1:
        n_train -= 1
    return shuffled[:n_train].tolist(), shuffled[n_train:].tolist()


# ── Loss and metrics ──────────────────────────────────────────────────────────

def re_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Huber(delta=0.05) + soft boundary barrier on Re_log_norm ∈ [0, 1].

    Huber caps gradient magnitude for large errors (|e| > 0.05), preventing
    BatchNorm destabilisation in early epochs when predictions may be 0.3–0.5
    normalised units from target. For |e| <= 0.05 it is quadratic like MSE,
    giving fine convergence at the end of training.

    pred / target shape: (B, 1)
    """
    huber   = F.huber_loss(pred, target, delta=0.05)
    barrier = (F.relu(pred - 1.0) ** 2 + F.relu(-pred) ** 2).mean()
    return huber + BARRIER_WEIGHT * barrier


def rmse_re_physical(
    preds_norm:   np.ndarray,
    targets_norm: np.ndarray,
) -> float:
    """Denormalise Re predictions and return RMSE in physical units."""
    pred_re = np.exp(
        np.clip(preds_norm, 0.0, 1.0) * (RE_LOG_MAX - RE_LOG_MIN) + RE_LOG_MIN
    )
    true_re = np.exp(targets_norm * (RE_LOG_MAX - RE_LOG_MIN) + RE_LOG_MIN)
    return float(np.sqrt(((pred_re - true_re) ** 2).mean()))


# ── Farthest Point Sampling (NumPy) ───────────────────────────────────────────

def fps_numpy(points: np.ndarray, npoint: int) -> np.ndarray:
    """
    Farthest Point Sampling on a (N, D) point array.
    Included for compatibility; unused by the CNN model.
    """
    N = points.shape[0]
    if N == 0:
        return np.zeros(npoint, dtype=np.int64)
    if N <= npoint:
        repeats = (npoint + N - 1) // N
        idx = np.tile(np.arange(N, dtype=np.int64), repeats)[:npoint]
        np.random.shuffle(idx)
        return idx
    centroids = np.empty(npoint, dtype=np.int64)
    distance  = np.full(N, np.inf, dtype=np.float64)
    farthest  = 0
    for i in range(npoint):
        centroids[i] = farthest
        centroid = points[farthest]
        dist = np.sum((points - centroid) ** 2, axis=-1)
        np.minimum(distance, dist, out=distance)
        farthest = int(np.argmax(distance))
    return centroids
