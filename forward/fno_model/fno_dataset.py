import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Normalization ranges for physics parameters (must match run_batch_openfoam_parallel.sh)
# These constants must be identical in: fno_dataset.py, infer_fno.py, eval_fno.py, vis.py
RE_MIN, RE_MAX = 100.0, 1000.0
UIN_MIN, UIN_MAX = 0.1, 1.0

def normalize_re(re):
    return (re - RE_MIN) / (RE_MAX - RE_MIN)

def normalize_uin(uin):
    return (uin - UIN_MIN) / (UIN_MAX - UIN_MIN)

def pad_to_efficient_grid(x, modes=None):
    """
    Pads spatial dimensions (last 3) to multiples of 4 for FFT efficiency.
    """
    *batch, D, H, W = x.shape

    def next_fast_len(n):
        m = 4
        return (n + m - 1) // m * m

    target_D = next_fast_len(D)
    target_H = next_fast_len(H)
    target_W = next_fast_len(W)

    pad_d = target_D - D
    pad_h = target_H - H
    pad_w = target_W - W

    # Pad only the end of each dimension: (left, right, top, bottom, front, back)
    return F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))


class CFDGridFnoDataset(Dataset):
    """
    Steady-state dataset for FNO-3D.

    Each case produced by simpleFoam contributes exactly ONE sample: the
    converged equilibrium velocity and pressure fields.

    Input (7 channels):
        0-2  : x_norm, y_norm, z_norm   (normalised spatial coordinates)
        3    : fluid_mask               (1 inside pipe, 0 outside)
        4    : cylinder_mask            (1 inside obstacle, 0 elsewhere)
        5    : Re_norm                  (Re normalised to [0, 1])
        6    : U_in_norm                (U_in normalised to [0, 1])

    Target (4 channels):
        0-2  : ux, uy, uz              (velocity / U_in)
        3    : p                       (pressure / U_in^2, mean-centred in fluid)

    Mask:
        (Nx, Ny, Nz) — 1 where the fluid is active and was sampled, 0 elsewhere.
    """

    def __init__(
        self,
        root_dir: str,
        metadata_csv: str = "metadata.csv",
        R_phys: float = 0.5,
        pad_data: bool = True,
        augment: bool = False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.R = R_phys
        self.pad_data = pad_data
        self.augment = augment

        meta_path = os.path.join(root_dir, metadata_csv)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata CSV not found: {meta_path}")

        self.cases = []
        with open(meta_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.cases.append(row)

        if not self.cases:
            raise RuntimeError("No CFD cases found in metadata.csv")

        # One sample per case — just collect valid case indices
        self.sample_index = []
        for ci, row in enumerate(self.cases):
            case_id = row["case_id"]
            case_dir = os.path.join(root_dir, case_id)
            if not os.path.isdir(case_dir):
                continue
            u_path = os.path.join(case_dir, "u.npy")
            p_path = os.path.join(case_dir, "p.npy")
            if not (os.path.isfile(u_path) and os.path.isfile(p_path)):
                continue
            self.sample_index.append(ci)

        self._case_cache = {}
        self._grid_cache = {}

    def __len__(self):
        return len(self.sample_index)

    def _load_case_arrays(self, case_idx):
        if case_idx in self._case_cache:
            return self._case_cache[case_idx]

        row = self.cases[case_idx]
        case_id = row["case_id"]
        case_dir = os.path.join(self.root_dir, case_id)

        coords = np.load(os.path.join(case_dir, "coords.npy"))
        u = np.load(os.path.join(case_dir, "u.npy"))   # (N_grid, 3)
        p = np.load(os.path.join(case_dir, "p.npy"))   # (N_grid,)

        mask_path = os.path.join(case_dir, "sampled_mask.npy")
        if os.path.isfile(mask_path):
            sampled_mask = np.load(mask_path).astype(np.float32)
        else:
            sampled_mask = np.ones((coords.shape[0],), dtype=np.float32)

        self._case_cache[case_idx] = (row, coords, u, p, sampled_mask)
        return self._case_cache[case_idx]

    def _get_grid_info(self, case_idx, coords, sampled_mask_flat):
        if case_idx in self._grid_cache:
            return self._grid_cache[case_idx]

        # Infer grid structure from unique coordinate values
        x_unique = np.unique(coords[:, 0])
        y_unique = np.unique(coords[:, 1])
        z_unique = np.unique(coords[:, 2])

        Nx, Ny, Nz = len(x_unique), len(y_unique), len(z_unique)

        Xg, Yg, Zg = np.meshgrid(x_unique, y_unique, z_unique, indexing="ij")

        # If coords are perfectly ordered x-major (as generated), use direct reshape
        if Nx * Ny * Nz == coords.shape[0]:
            idx_map = np.arange(Nx * Ny * Nz).reshape(Nx, Ny, Nz)
        else:
            # Fallback: build lookup map (slow, but cached per case)
            coord_to_idx = {tuple(c): i for i, c in enumerate(coords)}
            idx_map = np.zeros((Nx, Ny, Nz), dtype=int)
            for ix, xv in enumerate(x_unique):
                for iy, yv in enumerate(y_unique):
                    for iz, zv in enumerate(z_unique):
                        idx_map[ix, iy, iz] = coord_to_idx.get((xv, yv, zv), 0)

        # Normalised coordinates [0, 1]
        x_norm = (Xg - x_unique.min()) / (x_unique.max() - x_unique.min() + 1e-6)
        y_norm = (Yg - y_unique.min()) / (y_unique.max() - y_unique.min() + 1e-6)
        z_norm = (Zg - z_unique.min()) / (z_unique.max() - z_unique.min() + 1e-6)

        # Fluid mask: inside circular cross-section
        r = np.sqrt(Yg**2 + Zg**2)
        fluid_mask = (r <= self.R).astype(np.float32)

        # Cylinder/obstacle mask (approximate geometry: sphere at x≈3, r≈0.25)
        dist_c = np.sqrt((Xg - 3.0)**2 + Yg**2 + Zg**2)
        cylinder_mask = (dist_c <= 0.25).astype(np.float32)

        # OpenFOAM sampling mask mapped onto the 3D grid
        sm_grid = sampled_mask_flat[idx_map.reshape(-1)].reshape(Nx, Ny, Nz)

        # Training mask: inside fluid, outside obstacle, successfully sampled
        train_mask = fluid_mask * (1.0 - cylinder_mask) * sm_grid

        grid_tensors = (
            x_norm.astype(np.float32),
            y_norm.astype(np.float32),
            z_norm.astype(np.float32),
            fluid_mask,
            cylinder_mask,
            train_mask,
            idx_map,
        )
        self._grid_cache[case_idx] = (Nx, Ny, Nz, grid_tensors)
        return self._grid_cache[case_idx]

    def __getitem__(self, idx):
        case_idx = self.sample_index[idx]
        row, coords, u, p, sm_flat = self._load_case_arrays(case_idx)

        Nx, Ny, Nz, grid_tensors = self._get_grid_info(case_idx, coords, sm_flat)
        x_g, y_g, z_g, f_mask, c_mask, t_mask, idx_map = grid_tensors

        Re = float(row["Re"])
        U_in = float(row["U_in"])
        U0 = U_in  # velocity scale

        flat_idx = idx_map.reshape(-1)

        # Map flat grid arrays onto the 3D structured grid and normalise
        u_field = u[flat_idx].reshape(Nx, Ny, Nz, 3) / U0   # (Nx, Ny, Nz, 3)
        p_field = p[flat_idx].reshape(Nx, Ny, Nz) / (U0**2)  # (Nx, Ny, Nz)

        # Mean-centre pressure within the active fluid region
        mask_bool = t_mask > 0.5
        if mask_bool.any():
            p_field = p_field - np.mean(p_field[mask_bool])

        # 7-channel input: geometry + physics parameters (no current state)
        ones = np.ones_like(f_mask)
        in_stack = np.stack([
            x_g, y_g, z_g,                   # 0-2: normalised coordinates
            f_mask, c_mask,                   # 3-4: domain masks
            ones * normalize_re(Re),          # 5:   Reynolds number
            ones * normalize_uin(U_in),       # 6:   inlet velocity
        ], axis=0).astype(np.float32)         # (7, Nx, Ny, Nz)

        # 4-channel target: equilibrium velocity and pressure
        y_arr = np.stack([
            u_field[..., 0], u_field[..., 1], u_field[..., 2],
            p_field,
        ], axis=0).astype(np.float32)         # (4, Nx, Ny, Nz)

        x_ten = torch.from_numpy(in_stack)
        y_ten = torch.from_numpy(y_arr)
        m_ten = torch.from_numpy(t_mask.astype(np.float32))

        if self.pad_data:
            x_ten = pad_to_efficient_grid(x_ten.unsqueeze(0)).squeeze(0)
            y_ten = pad_to_efficient_grid(y_ten.unsqueeze(0)).squeeze(0)
            m_ten = pad_to_efficient_grid(m_ten.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        if self.augment:
            # Rotate 0, 90, 180, or 270 degrees around the pipe (x) axis.
            # Dims 2,3 of (C, Nx, Ny, Nz) are the y-z cross-section.
            # Velocity channels uy(1) and uz(2) must be rotated together with the field.
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                # Rotate spatial field
                x_ten = torch.rot90(x_ten, k=k, dims=(2, 3))
                y_ten = torch.rot90(y_ten, k=k, dims=(2, 3))
                m_ten = torch.rot90(m_ten, k=k, dims=(1, 2))
                # Rotate the uy/uz velocity vector components to match
                # 90° CCW: (uy, uz) → (-uz, uy)
                for _ in range(k):
                    uy = y_ten[1].clone()
                    uz = y_ten[2].clone()
                    y_ten[1] = -uz
                    y_ten[2] = uy

        return x_ten, y_ten, m_ten
