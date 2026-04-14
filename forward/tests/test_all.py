"""
Full test suite for the FNO steady-state pipeline.
Run from project root:  python -m pytest tests/test_all.py -v
"""

import sys
import os
import math
import tempfile
import csv

import numpy as np
import torch
import pytest

# Make project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fno_model.fno_model import FNO3d, SpectralConv3d
from fno_model.fno_dataset import (
    CFDGridFnoDataset,
    normalize_re,
    normalize_uin,
    pad_to_efficient_grid,
    RE_MIN, RE_MAX, UIN_MIN, UIN_MAX,
)

# Import loss from train.py
sys.path.insert(0, ROOT)
import importlib.util
spec = importlib.util.spec_from_file_location("train", os.path.join(ROOT, "train.py"))
_train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_train)
masked_relative_l2 = _train.masked_relative_l2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_model(modes_x=4, modes_y=4, modes_z=4, width=8, in_ch=7, out_ch=4):
    return FNO3d(
        modes_x=modes_x, modes_y=modes_y, modes_z=modes_z,
        width=width, in_channels=in_ch, out_channels=out_ch,
    )


def rand_input(B=2, C=7, Nx=16, Ny=8, Nz=8):
    return torch.randn(B, C, Nx, Ny, Nz)


# ---------------------------------------------------------------------------
# SpectralConv3d tests
# ---------------------------------------------------------------------------

class TestSpectralConv3d:

    def test_output_shape(self):
        conv = SpectralConv3d(in_channels=4, out_channels=8, modes_x=3, modes_y=3, modes_z=3)
        x = torch.randn(2, 4, 12, 8, 8)
        y = conv(x)
        assert y.shape == (2, 8, 12, 8, 8)

    def test_output_shape_asymmetric_modes(self):
        conv = SpectralConv3d(in_channels=4, out_channels=4, modes_x=6, modes_y=3, modes_z=3)
        x = torch.randn(2, 4, 16, 8, 8)
        y = conv(x)
        assert y.shape == (2, 4, 16, 8, 8)

    def test_output_is_real(self):
        conv = SpectralConv3d(in_channels=4, out_channels=4, modes_x=3, modes_y=3, modes_z=3)
        x = torch.randn(2, 4, 12, 8, 8)
        y = conv(x)
        assert y.dtype == torch.float32

    def test_weight_init_scale(self):
        """Weights should be initialised to Li et al. scale: 1/(in*out)."""
        in_ch, out_ch = 8, 8
        conv = SpectralConv3d(in_channels=in_ch, out_channels=out_ch,
                              modes_x=4, modes_y=4, modes_z=4)
        expected_max = 1.0 / (in_ch * out_ch)
        for name in ['weight1', 'weight2', 'weight3', 'weight4']:
            w = getattr(conv, name).data
            assert w.max().item() <= expected_max + 1e-9, \
                f"{name} max {w.max().item():.6f} exceeds scale {expected_max:.6f}"
            assert w.min().item() >= 0.0 - 1e-9, \
                f"{name} has negative values (should be uniform [0, scale])"

    def test_gradients_flow(self):
        conv = SpectralConv3d(in_channels=4, out_channels=4, modes_x=3, modes_y=3, modes_z=3)
        x = torch.randn(2, 4, 12, 8, 8, requires_grad=True)
        y = conv(x)
        y.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_modes_clipped_to_grid(self):
        """modes larger than the grid dimension should not crash."""
        conv = SpectralConv3d(in_channels=2, out_channels=2, modes_x=100, modes_y=100, modes_z=100)
        x = torch.randn(1, 2, 8, 8, 8)
        y = conv(x)
        assert y.shape == (1, 2, 8, 8, 8)

    def test_batch_size_one(self):
        conv = SpectralConv3d(in_channels=4, out_channels=4, modes_x=3, modes_y=3, modes_z=3)
        x = torch.randn(1, 4, 12, 8, 8)
        y = conv(x)
        assert y.shape == (1, 4, 12, 8, 8)

    def test_no_nan_in_output(self):
        conv = SpectralConv3d(in_channels=4, out_channels=4, modes_x=3, modes_y=3, modes_z=3)
        x = torch.randn(2, 4, 12, 8, 8)
        y = conv(x)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()


# ---------------------------------------------------------------------------
# FNO3d tests
# ---------------------------------------------------------------------------

class TestFNO3d:

    def test_output_shape(self):
        model = make_model()
        x = rand_input()
        y = model(x)
        assert y.shape == (2, 4, 16, 8, 8)

    def test_output_channels(self):
        model = make_model(out_ch=3)
        x = rand_input()
        y = model(x)
        assert y.shape[1] == 3

    def test_single_sample(self):
        model = make_model()
        x = rand_input(B=1)
        y = model(x)
        assert y.shape == (1, 4, 16, 8, 8)

    def test_no_nan_output(self):
        model = make_model()
        x = rand_input()
        y = model(x)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_gradients_end_to_end(self):
        model = make_model()
        x = rand_input()
        y = model(x)
        loss = y.sum()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No grad for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN grad for {name}"

    def test_in_channels_default(self):
        model = FNO3d()
        assert model.fc0.in_channels == 7

    def test_width_parameter(self):
        model = make_model(width=16)
        assert model.fc0.out_channels == 16

    def test_n_layers(self):
        model = FNO3d(modes_x=4, modes_y=4, modes_z=4, width=8, n_layers=3)
        assert len(model.convs) == 3
        assert len(model.ws) == 3

    def test_eval_mode_deterministic(self):
        model = make_model()
        model.eval()
        x = rand_input()
        with torch.no_grad():
            y1 = model(x)
            y2 = model(x)
        assert torch.allclose(y1, y2)

    def test_parameter_count_positive(self):
        model = make_model()
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n > 0

    def test_production_config_shape(self):
        """Verify the actual training config (modes 16/8/8, width 32) works."""
        model = FNO3d(modes_x=16, modes_y=8, modes_z=8, width=32, in_channels=7, out_channels=4)
        x = torch.randn(1, 7, 32, 8, 8)   # small spatial dims for speed
        y = model(x)
        assert y.shape == (1, 4, 32, 8, 8)


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

class TestMaskedRelativeL2:

    def _mask(self, shape):
        """Full mask (all ones)."""
        B, C, X, Y, Z = shape
        return torch.ones(B, X, Y, Z)

    def test_zero_loss_on_perfect_prediction(self):
        pred = torch.randn(2, 4, 8, 8, 8)
        mask = self._mask(pred.shape)
        loss = masked_relative_l2(pred, pred.clone(), mask)
        assert loss.item() < 1e-6

    def test_loss_is_positive(self):
        pred   = torch.randn(2, 4, 8, 8, 8)
        target = torch.randn(2, 4, 8, 8, 8)
        mask   = self._mask(pred.shape)
        loss = masked_relative_l2(pred, target, mask)
        assert loss.item() > 0

    def test_loss_is_scale_invariant(self):
        """Relative L2 should be identical regardless of global scale."""
        pred   = torch.randn(2, 4, 8, 8, 8)
        target = torch.randn(2, 4, 8, 8, 8)
        mask   = self._mask(pred.shape)
        l1 = masked_relative_l2(pred,       target,       mask).item()
        l2 = masked_relative_l2(pred * 100, target * 100, mask).item()
        assert abs(l1 - l2) < 1e-4, f"Loss not scale-invariant: {l1} vs {l2}"

    def test_mask_zeros_exclude_region(self):
        """Zero-masking the entire batch should give near-zero (denom ≈ 0 → clamped)."""
        pred   = torch.randn(2, 4, 8, 8, 8)
        target = torch.randn(2, 4, 8, 8, 8)
        mask   = torch.zeros(2, 8, 8, 8)
        loss = masked_relative_l2(pred, target, mask)
        # numerator = 0, denominator = 0, clamped → 0 / eps = 0
        assert loss.item() < 1e-3

    def test_partial_mask(self):
        """Loss with a partial mask should be finite and > 0."""
        pred   = torch.randn(2, 4, 8, 8, 8)
        target = torch.randn(2, 4, 8, 8, 8)
        mask   = torch.zeros(2, 8, 8, 8)
        mask[:, :4, :4, :4] = 1.0
        loss = masked_relative_l2(pred, target, mask)
        assert math.isfinite(loss.item())
        assert loss.item() > 0

    def test_4d_mask_auto_expanded(self):
        """mask of shape (B, X, Y, Z) should work without manual unsqueeze."""
        pred   = torch.randn(2, 4, 8, 8, 8)
        target = torch.randn(2, 4, 8, 8, 8)
        mask   = torch.ones(2, 8, 8, 8)
        loss = masked_relative_l2(pred, target, mask)
        assert math.isfinite(loss.item())

    def test_gradients_through_loss(self):
        pred   = torch.randn(2, 4, 8, 8, 8, requires_grad=True)
        target = torch.randn(2, 4, 8, 8, 8)
        mask   = self._mask(pred.shape)
        loss = masked_relative_l2(pred, target, mask)
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()

    def test_loss_bounded_reasonable(self):
        """For random pred/target the relative L2 should be O(1)."""
        torch.manual_seed(0)
        pred   = torch.randn(4, 4, 8, 8, 8)
        target = torch.randn(4, 4, 8, 8, 8)
        mask   = self._mask(pred.shape)
        loss = masked_relative_l2(pred, target, mask)
        assert 0.1 < loss.item() < 10.0, f"Unexpected loss magnitude: {loss.item()}"


# ---------------------------------------------------------------------------
# Normalisation / dataset utilities
# ---------------------------------------------------------------------------

class TestNormalization:

    def test_re_bounds(self):
        assert abs(normalize_re(RE_MIN)) < 1e-9
        assert abs(normalize_re(RE_MAX) - 1.0) < 1e-9

    def test_uin_bounds(self):
        assert abs(normalize_uin(UIN_MIN)) < 1e-9
        assert abs(normalize_uin(UIN_MAX) - 1.0) < 1e-9

    def test_re_midpoint(self):
        mid = (RE_MIN + RE_MAX) / 2
        assert abs(normalize_re(mid) - 0.5) < 1e-6

    def test_uin_midpoint(self):
        mid = (UIN_MIN + UIN_MAX) / 2
        assert abs(normalize_uin(mid) - 0.5) < 1e-6

    def test_normalize_re_in_range(self):
        for re in np.linspace(RE_MIN, RE_MAX, 20):
            v = normalize_re(re)
            assert 0.0 <= v <= 1.0 + 1e-9

    def test_normalize_uin_in_range(self):
        for uin in np.linspace(UIN_MIN, UIN_MAX, 20):
            v = normalize_uin(uin)
            assert 0.0 <= v <= 1.0 + 1e-9


class TestPadToEfficientGrid:

    def test_already_aligned(self):
        x = torch.zeros(1, 3, 8, 8, 8)
        out = pad_to_efficient_grid(x.unsqueeze(0)).squeeze(0)
        assert out.shape == x.shape

    def test_pads_to_multiple_of_4(self):
        x = torch.zeros(3, 7, 5, 6)   # last 3: 7,5,6 → 8,8,8
        out = pad_to_efficient_grid(x.unsqueeze(0)).squeeze(0)
        for d in out.shape[-3:]:
            assert d % 4 == 0

    def test_pad_values_are_zero(self):
        x = torch.ones(1, 7, 5, 5)
        out = pad_to_efficient_grid(x.unsqueeze(0)).squeeze(0)
        # Original values preserved
        assert (out[0, :7, :5, :5] == 1.0).all()
        # Padded values are zero
        if out.shape[-1] > 5:
            assert (out[0, :7, :5, 5:] == 0.0).all()

    def test_does_not_shrink(self):
        x = torch.zeros(2, 16, 20, 20)
        out = pad_to_efficient_grid(x.unsqueeze(0)).squeeze(0)
        for orig, padded in zip(x.shape[-3:], out.shape[-3:]):
            assert padded >= orig


# ---------------------------------------------------------------------------
# Dataset tests (using synthetic .npy files)
# ---------------------------------------------------------------------------

def make_synthetic_dataset(tmp_dir, n_cases=4, N=8, Nx=4, Ny=4, Nz=4):
    """Creates a minimal synthetic fno_grid directory structure."""
    # Build a tiny structured grid
    x_vals = np.linspace(0, 5, Nx)
    y_vals = np.linspace(-0.5, 0.5, Ny)
    z_vals = np.linspace(-0.5, 0.5, Nz)
    Xg, Yg, Zg = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    coords_flat = np.stack([Xg.ravel(), Yg.ravel(), Zg.ravel()], axis=1).astype(np.float32)
    N = len(coords_flat)

    rows = []
    for i in range(1, n_cases + 1):
        case_id = f"case_{i:04d}"
        case_dir = os.path.join(tmp_dir, case_id)
        os.makedirs(case_dir)

        Re  = 100.0 + (i - 1) * 200.0
        Uin = 0.1  + (i - 1) * 0.2
        nu  = Uin * 1.0 / Re

        np.save(os.path.join(case_dir, "coords.npy"), coords_flat)
        np.save(os.path.join(case_dir, "u.npy"),
                np.random.randn(N, 3).astype(np.float32) * Uin)
        np.save(os.path.join(case_dir, "p.npy"),
                np.random.randn(N).astype(np.float32) * Uin**2)
        np.save(os.path.join(case_dir, "sampled_mask.npy"),
                np.ones(N, dtype=np.float32))
        rows.append({"case_id": case_id, "Re": Re, "U_in": Uin, "nu": nu,
                     "R_pipe": 0.5, "D_pipe": 1.0})

    csv_path = os.path.join(tmp_dir, "metadata.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id","Re","U_in","nu","R_pipe","D_pipe"])
        writer.writeheader()
        writer.writerows(rows)

    return tmp_dir, N


class TestCFDGridFnoDataset:

    @pytest.fixture
    def ds(self, tmp_path):
        d, _ = make_synthetic_dataset(str(tmp_path), n_cases=4)
        return CFDGridFnoDataset(root_dir=d, metadata_csv="metadata.csv", pad_data=True)

    def test_length(self, ds):
        assert len(ds) == 4

    def test_getitem_shapes(self, ds):
        x, y, mask = ds[0]
        # 7 input channels; spatial dims padded to multiples of 4
        assert x.shape[0] == 7
        assert y.shape[0] == 4
        assert x.shape[1:] == y.shape[1:] == mask.shape

    def test_spatial_dims_multiple_of_4(self, ds):
        x, y, mask = ds[0]
        for d in x.shape[1:]:
            assert d % 4 == 0, f"Spatial dim {d} not divisible by 4"

    def test_input_coords_in_unit_range(self, ds):
        x, y, mask = ds[0]
        for ch in range(3):   # x_norm, y_norm, z_norm
            assert x[ch].min() >= -1e-5
            assert x[ch].max() <= 1.0 + 1e-5

    def test_re_channel_constant(self, ds):
        """Re and U_in channels should be spatially constant."""
        x, _, _ = ds[0]
        for ch in [5, 6]:
            vals = x[ch]
            assert vals.std().item() < 1e-6, f"Channel {ch} is not spatially constant"

    def test_mask_binary(self, ds):
        _, _, mask = ds[0]
        unique = mask.unique()
        for v in unique:
            assert v.item() in (0.0, 1.0), f"Unexpected mask value {v.item()}"

    def test_target_finite(self, ds):
        _, y, _ = ds[0]
        assert torch.isfinite(y).all()

    def test_pressure_mean_centred(self, ds):
        """Pressure (channel 3 of target) should be ~zero-mean in the fluid."""
        _, y, mask = ds[0]
        p = y[3]
        fluid = mask.bool()
        if fluid.any():
            mean_p = p[fluid].mean().item()
            assert abs(mean_p) < 0.1, f"Pressure not mean-centred: {mean_p}"

    def test_no_padding_option(self, tmp_path):
        d, _ = make_synthetic_dataset(str(tmp_path), n_cases=2)
        ds = CFDGridFnoDataset(root_dir=d, metadata_csv="metadata.csv", pad_data=False)
        x, y, mask = ds[0]
        # spatial dims should be exactly Nx×Ny×Nz = 4×4×4
        assert x.shape == (7, 4, 4, 4)
        assert y.shape == (4, 4, 4, 4)

    def test_all_cases_loadable(self, ds):
        for i in range(len(ds)):
            x, y, mask = ds[i]
            assert torch.isfinite(x).all()
            assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# Training sanity check
# ---------------------------------------------------------------------------

class TestTrainingSanity:

    def test_loss_decreases_over_steps(self, tmp_path):
        """Loss should strictly decrease over a few gradient steps on a tiny batch."""
        torch.manual_seed(42)
        model = make_model(modes_x=2, modes_y=2, modes_z=2, width=4)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Tiny fixed batch
        B, C_in, Nx, Ny, Nz = 2, 7, 8, 8, 8
        x      = torch.randn(B, C_in, Nx, Ny, Nz)
        target = torch.randn(B, 4, Nx, Ny, Nz)
        mask   = torch.ones(B, Nx, Ny, Nz)

        losses = []
        for _ in range(10):
            opt.zero_grad()
            pred = model(x)
            loss = masked_relative_l2(pred, target, mask)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], \
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_model_can_overfit_single_sample(self, tmp_path):
        """Model should be able to drive loss close to 0 on one sample."""
        torch.manual_seed(0)
        model = make_model(modes_x=4, modes_y=4, modes_z=4, width=16)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-2)

        x      = torch.randn(1, 7, 8, 8, 8)
        target = torch.randn(1, 4, 8, 8, 8)
        mask   = torch.ones(1, 8, 8, 8)

        for _ in range(200):
            opt.zero_grad()
            pred = model(x)
            loss = masked_relative_l2(pred, target, mask)
            loss.backward()
            opt.step()

        assert loss.item() < 0.05, \
            f"Failed to overfit single sample: loss = {loss.item():.4f}"

    def test_dataset_integrates_with_dataloader(self, tmp_path):
        from torch.utils.data import DataLoader
        d, _ = make_synthetic_dataset(str(tmp_path), n_cases=4)
        ds = CFDGridFnoDataset(root_dir=d, metadata_csv="metadata.csv", pad_data=True)
        loader = DataLoader(ds, batch_size=2)
        x, y, mask = next(iter(loader))
        assert x.shape[0] == 2
        assert y.shape[0] == 2
        assert mask.shape[0] == 2

    def test_loss_zero_for_perfect_pred_from_model(self):
        """If pred == target, loss should be 0 regardless of model."""
        model = make_model()
        model.eval()
        with torch.no_grad():
            x = rand_input()
            pred = model(x)
        mask = torch.ones(pred.shape[0], *pred.shape[2:])
        loss = masked_relative_l2(pred, pred, mask)
        assert loss.item() < 1e-6
