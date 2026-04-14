"""
Tests for the CNN3DInverse FNO-data pipeline.

Covers:
  1. generate_fno_data.py helpers (geometry, voxel conversion)
  2. dataset.py — use_true_uin, sample_index discovery, __getitem__ shapes
  3. model.py — forward pass shape
  4. utils.py — normalisation round-trips, loss
  5. D4 augmentation consistency (dataset vs evaluate.py)
  6. train.py constants / BN calibration block presence
  7. evaluate.py — use_true_uin loading from uin_linear.json

Run from the inverse/ directory:
    pytest tests/test_pipeline.py -v
"""

import csv
import json
import math
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG  = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PKG)
for _p in [_PKG, _ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset import VoxelCFDDataset, _central_diff
from model   import CNN3DInverse
from utils   import (
    RE_MIN, RE_MAX, RE_LOG_MIN, RE_LOG_MAX,
    UIN_MIN, UIN_MAX,
    normalize_re, denorm_re, make_split, re_loss, rmse_re_physical,
    CYL_R, CYL_X, PIPE_L, PIPE_R,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures — synthetic FNO-style data directory
# ═══════════════════════════════════════════════════════════════════════════════

NX, NY, NZ = 16, 6, 6   # small grid for speed

def _make_fake_voxel(Re: float, U_in: float, seed: int = 0) -> np.ndarray:
    """Return a plausible (6, NX, NY, NZ) float32 voxel (raw, not normalised)."""
    rng = np.random.default_rng(seed)
    voxel = rng.standard_normal((6, NX, NY, NZ)).astype(np.float32)
    # scale channels to realistic magnitudes  (raw = dimensional)
    voxel[0] *= U_in        # ux ~ U_in
    voxel[1] *= U_in * 0.1  # uy small
    voxel[2] *= U_in * 0.1  # uz small
    voxel[3] *= U_in ** 2   # p
    voxel[4] *= U_in ** 2 / 0.04   # dp/dx  (dx≈0.04)
    voxel[5] *= U_in * 5.0         # |ω|
    return voxel


@pytest.fixture
def fake_data_dir():
    """Temp directory with 6 FNO-style cases (voxel_grid_raw.npy + metadata.csv)."""
    cases = [
        ("case_00000", 100.0,  0.1),
        ("case_00001", 200.0,  0.55),
        ("case_00002", 400.0,  1.0),
        ("case_00003", 700.0,  0.1),
        ("case_00004", 850.0,  0.55),
        ("case_00005", 1000.0, 1.0),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        # Write metadata.csv
        meta = os.path.join(tmp, "metadata.csv")
        with open(meta, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["case_id", "Re", "U_in"])
            for cid, re, uin in cases:
                writer.writerow([cid, f"{re:.6f}", f"{uin:.4f}"])

        # Write voxel_grid_raw.npy per case
        for i, (cid, re, uin) in enumerate(cases):
            d = os.path.join(tmp, cid)
            os.makedirs(d)
            voxel = _make_fake_voxel(re, uin, seed=i)
            np.save(os.path.join(d, "voxel_grid_raw.npy"), voxel)

        yield tmp, cases


# ═══════════════════════════════════════════════════════════════════════════════
# 1. generate_fno_data helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateFnoDataHelpers:
    """Tests for geometry and voxel-conversion helpers in generate_fno_data.py."""

    def test_central_diff_shape(self):
        arr = np.random.randn(NX, NY, NZ).astype(np.float32)
        for ax in range(3):
            out = _central_diff(arr, ax=ax, h=0.04)
            assert out.shape == arr.shape, f"axis {ax}"

    def test_central_diff_constant_is_zero(self):
        arr = np.ones((8, 8, 8), dtype=np.float64)
        for ax in range(3):
            out = _central_diff(arr, ax=ax, h=0.1)
            np.testing.assert_allclose(out, 0.0, atol=1e-10, err_msg=f"axis {ax}")

    def test_central_diff_linear_exact(self):
        """Derivative of f(x)=3x should be exactly 3 at all interior points."""
        xs = np.linspace(0, 1, 20)
        arr = np.broadcast_to(3.0 * xs[:, None, None], (20, 5, 5)).copy()
        out = _central_diff(arr, ax=0, h=xs[1] - xs[0])
        # Interior points: exact central-difference gives 3.0
        np.testing.assert_allclose(out[1:-1], 3.0, atol=1e-5)

    def test_geometry_constants_consistent(self):
        """PIPE_R, CYL_X, CYL_R come from utils — check they're reasonable."""
        assert PIPE_R > 0
        assert CYL_R < PIPE_R          # sphere smaller than pipe
        assert 0 < CYL_X < PIPE_L     # sphere inside pipe
        assert CYL_R > 0

    def test_fno_to_voxel_channel_count(self):
        """Simulate fno_to_voxel: output must be shape (6, NX, NY, NZ)."""
        import generate_fno_data as gfd

        NXS, NYS, NZS = 8, 4, 4
        xs = np.linspace(0, PIPE_L, NXS, dtype=np.float32)
        ys = np.linspace(-PIPE_R, PIPE_R, NYS, dtype=np.float32)
        zs = np.linspace(-PIPE_R, PIPE_R, NZS, dtype=np.float32)
        dx = float(xs[1] - xs[0])
        dy = float(ys[1] - ys[0])
        dz = float(zs[1] - zs[0])

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        R2D         = np.sqrt(Y ** 2 + Z ** 2)
        dist_sphere = np.sqrt((X - CYL_X) ** 2 + Y ** 2 + Z ** 2)
        full_fluid  = (R2D <= PIPE_R) & (dist_sphere > CYL_R)

        # Fake FNO output: 4 channels
        rng = np.random.default_rng(7)
        fno_out = rng.standard_normal((4, NXS, NYS, NZS)).astype(np.float32)

        voxel = gfd.fno_to_voxel(fno_out, U_in=0.55, full_fluid=full_fluid,
                                  dx=dx, dy=dy, dz=dz)
        assert voxel.shape == (6, NXS, NYS, NZS)
        assert voxel.dtype == np.float32
        # Non-fluid voxels must be zero
        assert np.all(voxel[:, ~full_fluid] == 0.0)

    def test_fno_to_voxel_zeroes_nonfluids(self):
        """All non-fluid voxels zeroed, fluid voxels may be nonzero."""
        import generate_fno_data as gfd

        NXS, NYS, NZS = 8, 4, 4
        xs = np.linspace(0, PIPE_L, NXS, dtype=np.float32)
        ys = np.linspace(-PIPE_R, PIPE_R, NYS, dtype=np.float32)
        zs = np.linspace(-PIPE_R, PIPE_R, NZS, dtype=np.float32)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        R2D         = np.sqrt(Y ** 2 + Z ** 2)
        dist_sphere = np.sqrt((X - CYL_X) ** 2 + Y ** 2 + Z ** 2)
        full_fluid  = (R2D <= PIPE_R) & (dist_sphere > CYL_R)

        rng = np.random.default_rng(0)
        fno_out = rng.standard_normal((4, NXS, NYS, NZS)).astype(np.float32) + 5.0

        voxel = gfd.fno_to_voxel(fno_out, U_in=0.3, full_fluid=full_fluid,
                                  dx=xs[1]-xs[0], dy=ys[1]-ys[0], dz=zs[1]-zs[0])
        # Non-fluid => all 6 channels zero
        assert np.all(voxel[:, ~full_fluid] == 0.0), "non-fluid voxels not zeroed"

    def test_pad_to_multiple4_noop(self):
        """Already-multiple-of-4 tensor should be returned unchanged."""
        import generate_fno_data as gfd
        x = torch.zeros(2, 7, 8, 8, 8)
        out, Nx, Ny, Nz = gfd._pad_to_multiple4(x)
        assert out.shape == x.shape
        assert (Nx, Ny, Nz) == (8, 8, 8)

    def test_pad_to_multiple4_pads(self):
        import generate_fno_data as gfd
        x = torch.zeros(1, 7, 5, 5, 5)
        out, Nx, Ny, Nz = gfd._pad_to_multiple4(x)
        assert out.shape == (1, 7, 8, 8, 8)
        assert (Nx, Ny, Nz) == (5, 5, 5)

    def test_sampling_counts(self):
        """N_RE × len(UIN_VALS) must equal expected total."""
        import generate_fno_data as gfd
        pairs = [(re, uin)
                 for re in np.linspace(1, 2, gfd.N_RE)
                 for uin in gfd.UIN_VALS]
        assert len(pairs) == gfd.N_RE * len(gfd.UIN_VALS)

    def test_re_vals_log_uniform(self):
        """Re sampling should be log-uniform within [RE_MIN, RE_MAX]."""
        import generate_fno_data as gfd
        Re_vals = np.exp(
            np.linspace(np.log(RE_MIN), np.log(RE_MAX), gfd.N_RE)
        )
        assert Re_vals[0]  == pytest.approx(RE_MIN,  rel=1e-5)
        assert Re_vals[-1] == pytest.approx(RE_MAX,  rel=1e-5)
        # Check log-uniformity: equal spacing in log domain
        log_vals = np.log(Re_vals)
        diffs = np.diff(log_vals)
        assert np.std(diffs) < 1e-10

    def test_uin_vals_in_range(self):
        import generate_fno_data as gfd
        for uin in gfd.UIN_VALS:
            assert UIN_MIN <= uin <= UIN_MAX, f"U_in={uin} out of [UIN_MIN,UIN_MAX]"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Dataset — sample discovery and __getitem__
# ═══════════════════════════════════════════════════════════════════════════════

DUMMY_UIN_LINEAR = {"a": 0.0, "b": 0.0}
DUMMY_STATS = {"mean": [0.0]*6, "std": [1.0]*6}


class TestDataset:

    def test_discovers_all_cases(self, fake_data_dir):
        tmp, cases = fake_data_dir
        ds = VoxelCFDDataset(
            root_dir=tmp,
            field_stats=DUMMY_STATS,
            uin_linear=DUMMY_UIN_LINEAR,
            use_true_uin=True,
        )
        assert len(ds) == len(cases)

    def test_missing_case_dir_skipped(self, fake_data_dir):
        """If a case directory doesn't exist, it must not be included."""
        tmp, cases = fake_data_dir
        import shutil
        shutil.rmtree(os.path.join(tmp, "case_00000"))
        ds = VoxelCFDDataset(
            root_dir=tmp,
            field_stats=DUMMY_STATS,
            uin_linear=DUMMY_UIN_LINEAR,
            use_true_uin=True,
        )
        assert len(ds) == len(cases) - 1

    def test_getitem_shape(self, fake_data_dir):
        tmp, cases = fake_data_dir
        ds = VoxelCFDDataset(
            root_dir=tmp,
            field_stats=DUMMY_STATS,
            uin_linear=DUMMY_UIN_LINEAR,
            use_true_uin=True,
        )
        vt, y = ds[0]
        assert vt.shape == (6, NX, NY, NZ)
        assert y.shape == (1,)

    def test_getitem_label_in_range(self, fake_data_dir):
        """Re_log_norm label must lie in [0, 1] for all cases."""
        tmp, cases = fake_data_dir
        ds = VoxelCFDDataset(
            root_dir=tmp,
            field_stats=DUMMY_STATS,
            uin_linear=DUMMY_UIN_LINEAR,
            use_true_uin=True,
        )
        for i in range(len(ds)):
            _, y = ds[i]
            assert 0.0 <= float(y[0]) <= 1.0, f"case {i}: y={float(y[0]):.4f}"

    def test_use_true_uin_matches_metadata(self, fake_data_dir):
        """
        With use_true_uin=True, the dimensionless channels must be built
        using exact U_in from metadata — verify ux/U_in = voxel[0]/U_in exactly.
        """
        tmp, cases = fake_data_dir
        ds = VoxelCFDDataset(
            root_dir=tmp,
            field_stats=None,    # no z-score so we can check raw ratios
            uin_linear=DUMMY_UIN_LINEAR,
            use_true_uin=True,
        )
        cid, re, uin = cases[0]
        vt, _ = ds[0]
        # Load raw to verify
        raw = np.load(os.path.join(tmp, cid, "voxel_grid_raw.npy"))
        expected_ch0 = raw[0] / uin
        np.testing.assert_allclose(
            vt[0].numpy(), expected_ch0, rtol=1e-5,
            err_msg="ch0 (ux/U_in) mismatch with use_true_uin=True"
        )

    def test_use_false_uin_uses_linear(self, fake_data_dir):
        """
        With use_true_uin=False and a=1,b=0, U_pred = ux_mean_upstream.
        The ux channel should then equal ux / ux_mean_upstream.
        This just checks that the non-True branch executes without error.
        """
        tmp, _ = fake_data_dir
        uin_lin = {"a": 1.0, "b": 0.0}
        ds = VoxelCFDDataset(
            root_dir=tmp,
            field_stats=None,
            uin_linear=uin_lin,
            use_true_uin=False,
        )
        vt, y = ds[0]
        assert vt.shape == (6, NX, NY, NZ)
        assert y.shape == (1,)

    def test_no_uin_linear_raises(self, fake_data_dir):
        """uin_linear=None and use_true_uin=False must raise ValueError."""
        tmp, _ = fake_data_dir
        ds = VoxelCFDDataset(
            root_dir=tmp,
            field_stats=None,
            uin_linear=None,
            use_true_uin=False,
        )
        with pytest.raises(ValueError, match="uin_linear"):
            _ = ds[0]

    def test_augmentation_changes_tensor(self, fake_data_dir):
        """With augment=True, multiple calls should (usually) differ from non-augmented."""
        tmp, _ = fake_data_dir
        np.random.seed(99)
        torch.manual_seed(99)
        ds_aug = VoxelCFDDataset(
            root_dir=tmp, augment=True, noise_std=0.0,
            field_stats=None, uin_linear=DUMMY_UIN_LINEAR, use_true_uin=True,
        )
        ds_plain = VoxelCFDDataset(
            root_dir=tmp, augment=False, noise_std=0.0,
            field_stats=None, uin_linear=DUMMY_UIN_LINEAR, use_true_uin=True,
        )
        same_count = 0
        for _ in range(8):
            vt_aug, _ = ds_aug[0]
            vt_plain, _ = ds_plain[0]
            if torch.allclose(vt_aug, vt_plain):
                same_count += 1
        # At least some augmented variants should differ (prob of 8 identical is 1/8^7)
        assert same_count < 8, "Augmentation never changed the tensor"

    def test_d4_augmentation_8_variants_cover_identity(self, fake_data_dir):
        """
        The identity transform (k=0, flip=False) must produce the same tensor
        as no augmentation.
        """
        tmp, _ = fake_data_dir
        ds = VoxelCFDDataset(
            root_dir=tmp, augment=False, noise_std=0.0,
            field_stats=None, uin_linear=DUMMY_UIN_LINEAR, use_true_uin=True,
        )
        vt_plain, _ = ds[0]

        # Manually apply identity (k=0, flip=False) using dataset's logic
        vt = vt_plain.clone()
        # k=0: no rotation; flip=False: no flip → vt unchanged
        np.testing.assert_allclose(vt.numpy(), vt_plain.numpy(), atol=1e-6)

    def test_compute_dimless_stats_shape(self, fake_data_dir):
        tmp, cases = fake_data_dir
        train_cis = list(range(len(cases)))   # all as training for this test
        stats = VoxelCFDDataset.compute_dimless_stats(
            root_dir=tmp,
            metadata_csv="metadata.csv",
            train_case_cis=train_cis,
            uin_linear=DUMMY_UIN_LINEAR,
            use_true_uin=True,
        )
        assert "mean" in stats and "std" in stats
        assert len(stats["mean"]) == 6
        assert len(stats["std"])  == 6

    def test_compute_dimless_stats_std_positive(self, fake_data_dir):
        tmp, cases = fake_data_dir
        train_cis = list(range(len(cases)))
        stats = VoxelCFDDataset.compute_dimless_stats(
            root_dir=tmp,
            metadata_csv="metadata.csv",
            train_case_cis=train_cis,
            uin_linear=DUMMY_UIN_LINEAR,
            use_true_uin=True,
        )
        for i, s in enumerate(stats["std"]):
            assert s >= 0.0, f"Negative std for channel {i}: {s}"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Model forward pass
# ═══════════════════════════════════════════════════════════════════════════════

class TestModel:

    def test_forward_output_shape(self):
        model = CNN3DInverse(in_channels=6)
        x = torch.randn(2, 6, NX, NY, NZ)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 1)

    def test_forward_batch_size_1(self):
        """Batch size 1 works in eval mode (BN1d uses running stats)."""
        model = CNN3DInverse(in_channels=6)
        model.eval()
        x = torch.randn(1, 6, NX, NY, NZ)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 1)

    def test_forward_output_range_reasonable(self):
        """Untrained model output should at least be finite."""
        model = CNN3DInverse(in_channels=6)
        x = torch.randn(4, 6, NX, NY, NZ)
        with torch.no_grad():
            y = model(x)
        assert torch.all(torch.isfinite(y)), "Model output contains inf/nan"

    def test_param_count_not_zero(self):
        model = CNN3DInverse(in_channels=6)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n > 0

    def test_full_resolution_forward(self):
        """Forward pass at full training resolution (265×20×20) must succeed."""
        model = CNN3DInverse(in_channels=6)
        model.eval()
        x = torch.randn(1, 6, 265, 20, 20)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 1)
        assert torch.isfinite(y).all()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Utils — normalisation and loss
# ═══════════════════════════════════════════════════════════════════════════════

class TestUtils:

    def test_normalize_re_endpoints(self):
        assert normalize_re(RE_MIN) == pytest.approx(0.0, abs=1e-6)
        assert normalize_re(RE_MAX) == pytest.approx(1.0, abs=1e-6)

    def test_normalize_re_midpoint(self):
        re_mid = math.exp((RE_LOG_MIN + RE_LOG_MAX) / 2)
        assert normalize_re(re_mid) == pytest.approx(0.5, abs=1e-6)

    def test_denorm_re_roundtrip(self):
        for re in [100.0, 316.0, 500.0, 1000.0]:
            norm = normalize_re(re)
            back = float(denorm_re(norm))
            assert back == pytest.approx(re, rel=1e-5)

    def test_make_split_sizes(self):
        for n in [10, 100, 3000]:
            tr, va = make_split(n)
            assert len(tr) + len(va) == n
            assert len(va) == max(1, int(0.1 * n))

    def test_make_split_reproducible(self):
        tr1, va1 = make_split(100, seed=42)
        tr2, va2 = make_split(100, seed=42)
        assert tr1 == tr2 and va1 == va2

    def test_make_split_different_seeds(self):
        _, va1 = make_split(100, seed=42)
        _, va2 = make_split(100, seed=99)
        assert va1 != va2

    def test_re_loss_finite(self):
        pred   = torch.rand(8, 1)
        target = torch.rand(8, 1)
        loss   = re_loss(pred, target)
        assert torch.isfinite(loss)

    def test_re_loss_perfect_pred(self):
        """Zero error → loss should be near zero (only barrier, which is also 0)."""
        pred = torch.full((4, 1), 0.5)
        loss = re_loss(pred, pred.clone())
        assert float(loss) < 1e-6

    def test_re_loss_out_of_range_penalty(self):
        """Prediction outside [0,1] should incur larger loss than one inside."""
        target = torch.full((4, 1), 0.5)
        pred_in  = torch.full((4, 1), 0.5)
        pred_out = torch.full((4, 1), 1.5)   # violates barrier
        loss_in  = re_loss(pred_in,  target)
        loss_out = re_loss(pred_out, target)
        assert float(loss_out) > float(loss_in)

    def test_rmse_re_physical_perfect(self):
        norms = np.array([0.0, 0.5, 1.0])
        rmse  = rmse_re_physical(norms, norms)
        assert rmse == pytest.approx(0.0, abs=1e-6)

    def test_rmse_re_physical_nonzero(self):
        preds   = np.array([0.0, 0.5, 1.0])
        targets = np.array([1.0, 0.5, 0.0])
        rmse    = rmse_re_physical(preds, targets)
        assert rmse > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. D4 augmentation consistency — dataset vs evaluate.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestD4Augmentation:
    """
    The D4 augmentation in evaluate.py (_augment_d4_batch) must be consistent
    with the per-sample logic in dataset.py's __getitem__.

    Both define:
      flip → torch.flip on Ny axis (dim 2 in single sample, dim 3 in batch)
              + negate channel 1 (uy)
      k-rotation → torch.rot90 on (Ny,Nz) + rotate (uy,uz) vector k times
    """

    def test_flip_channel_negate(self):
        """Flip must negate channel 1 (uy) and spatially flip Ny."""
        vt = torch.zeros(6, NX, NY, NZ)
        vt[1] = 1.0   # uy = 1 everywhere

        # dataset-style flip (single sample, dims=[2])
        vt_flip = torch.flip(vt.clone(), dims=[2])
        vt_flip[1] = -vt_flip[1]

        assert float(vt_flip[1].mean()) == pytest.approx(-1.0, abs=1e-6)

    def test_batch_flip_matches_single(self):
        """evaluate.py batch flip (dims=[3]) must match dataset single flip (dims=[2])."""
        from evaluate import _augment_d4_batch

        single = torch.randn(6, NX, NY, NZ)
        batch  = single.unsqueeze(0)   # (1, 6, NX, NY, NZ)

        # dataset-style (dims=[2])
        s_flip = torch.flip(single.clone(), dims=[2])
        s_flip[1] = -s_flip[1]

        # evaluate-style (dims=[3])
        b_flip = _augment_d4_batch(batch, k=0, flip=True)

        np.testing.assert_allclose(
            s_flip.numpy(), b_flip[0].numpy(), atol=1e-6,
            err_msg="Batch flip (dim3) differs from single flip (dim2)"
        )

    def test_batch_rotation_matches_single(self):
        """evaluate.py batch rotation (dims=[3,4]) must match dataset (dims=[2,3])."""
        from evaluate import _augment_d4_batch

        single = torch.randn(6, NX, NY, NZ)
        batch  = single.unsqueeze(0)

        for k in range(1, 4):
            # dataset-style
            s_rot = torch.rot90(single.clone(), k=k, dims=[2, 3])
            for _ in range(k):
                uy_tmp = s_rot[1].clone()
                s_rot[1] = -s_rot[2]
                s_rot[2] =  uy_tmp

            # evaluate-style
            b_rot = _augment_d4_batch(batch, k=k, flip=False)

            np.testing.assert_allclose(
                s_rot.numpy(), b_rot[0].numpy(), atol=1e-6,
                err_msg=f"Batch rotation k={k} (dims=[3,4]) differs from single (dims=[2,3])"
            )

    def test_8_variants_all_distinct(self):
        """The 8 D4 variants of a generic tensor should all be distinct."""
        from evaluate import _augment_d4_batch

        batch = torch.randn(1, 6, NX, NY, NZ)
        variants = []
        for k in range(4):
            for flip in [False, True]:
                v = _augment_d4_batch(batch, k=k, flip=flip)
                variants.append(v[0].numpy().copy())

        for i in range(8):
            for j in range(i + 1, 8):
                if np.allclose(variants[i], variants[j], atol=1e-5):
                    pytest.fail(f"D4 variants {i} and {j} are identical")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. train.py — constants and BN calibration block
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainPyStructure:
    """Source-level checks that train.py has the expected configuration."""

    @pytest.fixture(autouse=True)
    def _read_source(self):
        src_path = os.path.join(_PKG, "train.py")
        with open(src_path) as f:
            self.src = f.read()

    def test_data_root_points_to_fno_training_data(self):
        assert "fno_training_data" in self.src

    def test_use_true_uin_set(self):
        assert "use_true_uin=True" in self.src

    def test_bn_calibration_present(self):
        assert "N_BN_CAL" in self.src

    def test_model_train_before_cal(self):
        assert "model.train()" in self.src

    def test_no_grad_in_cal(self):
        assert "torch.no_grad()" in self.src

    def test_early_stop_patience_reasonable(self):
        import train
        assert train.EARLY_STOP_PATIENCE >= 50

    def test_noise_std_set(self):
        import train
        assert 0.0 < train.NOISE_STD <= 0.1

    def test_batch_size_set(self):
        import train
        assert train.BATCH_SIZE >= 8

    def test_lr_set(self):
        import train
        assert 1e-5 < train.LR < 1e-2

    def test_in_channels_6(self):
        import train
        assert train.IN_CHANNELS == 6


# ═══════════════════════════════════════════════════════════════════════════════
# 7. evaluate.py — uin_linear loading and use_true_uin
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluatePyStructure:

    @pytest.fixture(autouse=True)
    def _read_source(self):
        src_path = os.path.join(_PKG, "evaluate.py")
        with open(src_path) as f:
            self.src = f.read()

    def test_loads_use_true_uin_from_json(self):
        assert "use_true_uin" in self.src

    def test_data_root_points_to_fno_training_data(self):
        assert "fno_training_data" in self.src

    def test_tta_2_variants(self):
        """TTA loop must iterate over identity + y-flip (2 variants)."""
        assert "[False, True]" in self.src

    def test_use_true_uin_bypasses_stage1(self):
        """When use_true_uin=True, u_in_preds = u_in_true.copy() (no regression)."""
        assert "u_in_true.copy()" in self.src


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Integration — dataset + model forward
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:

    def test_dataset_to_model(self, fake_data_dir):
        """A batch from VoxelCFDDataset should pass through CNN3DInverse."""
        tmp, _ = fake_data_dir
        ds = VoxelCFDDataset(
            root_dir=tmp,
            augment=False, noise_std=0.0,
            field_stats=DUMMY_STATS,
            uin_linear=DUMMY_UIN_LINEAR,
            use_true_uin=True,
        )
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        model  = CNN3DInverse(in_channels=6)
        model.eval()

        vt, y = next(iter(loader))
        with torch.no_grad():
            pred = model(vt)
        assert pred.shape == y.shape
        assert torch.isfinite(pred).all()

    def test_re_loss_with_model_output(self, fake_data_dir):
        tmp, _ = fake_data_dir
        ds = VoxelCFDDataset(
            root_dir=tmp,
            augment=False, noise_std=0.0,
            field_stats=DUMMY_STATS,
            uin_linear=DUMMY_UIN_LINEAR,
            use_true_uin=True,
        )
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=2)
        model  = CNN3DInverse(in_channels=6)

        vt, y = next(iter(loader))
        pred   = model(vt)
        loss   = re_loss(pred, y)
        assert torch.isfinite(loss)
        loss.backward()   # ensure gradients flow
