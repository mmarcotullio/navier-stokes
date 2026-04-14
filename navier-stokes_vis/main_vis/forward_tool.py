# forward_tool.py
# Inference-relevant code copied from forward/fno_model/
# (SpectralConv3d, FNO3d from fno_model.py; helpers from fno_dataset.py)

import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

# ── Normalization ranges (must match training) ────────────────────────────────
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


# ── SpectralConv3d ────────────────────────────────────────────────────────────

class SpectralConv3d(nn.Module):
    """
    3D Fourier spectral convolution layer.

    Input:  (B, C_in, Nx, Ny, Nz)
    Output: (B, C_out, Nx, Ny, Nz)

    We keep only the lowest `modes_*` Fourier modes in each dimension.
    """

    def __init__(self, in_channels, out_channels,
                 modes_x, modes_y, modes_z):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z

        # Match Li et al. (2020) initialisation: uniform in [0, 1/(in*out)]
        scale = 1.0 / (in_channels * out_channels)
        shape = (out_channels, in_channels, self.modes_x, self.modes_y, self.modes_z, 2)
        self.weight1 = nn.Parameter(scale * torch.rand(shape))
        self.weight2 = nn.Parameter(scale * torch.rand(shape))
        self.weight3 = nn.Parameter(scale * torch.rand(shape))
        self.weight4 = nn.Parameter(scale * torch.rand(shape))

    def compl_mul3d(self, input, weights):
        # (B, C_in, Nx, Ny, Nz/2+1), (C_out, C_in, mx, my, mz, 2)
        # We only use the low-frequency block; we assume input is already sliced.
        # input: complex tensor
        # weights: real+imag
        wr = weights[..., 0]
        wi = weights[..., 1]
        # input: (B, C_in, mx, my, mz)
        xr = input.real
        xi = input.imag
        # (B, C_out, mx, my, mz) via complex matmul over channel dimension
        out_real = torch.einsum("bixyz, oixyz -> boxyz", xr, wr) - torch.einsum("bixyz, oixyz -> boxyz", xi, wi)
        out_imag = torch.einsum("bixyz, oixyz -> boxyz", xr, wi) + torch.einsum("bixyz, oixyz -> boxyz", xi, wr)
        return torch.complex(out_real, out_imag)

    def forward(self, x):
        """
        x: (B, C_in, Nx, Ny, Nz)
        """
        B, C, Nx, Ny, Nz = x.shape

        # Forward FFT
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))  # (B, C_in, Nx, Ny, Nz//2+1)

        # Allocate output FFT tensor
        out_ft = torch.zeros(
            B,
            self.out_channels,
            Nx,
            Ny,
            Nz // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        mx = min(self.modes_x, Nx)
        my = min(self.modes_y, Ny)
        mz = min(self.modes_z, Nz // 2 + 1)

        # All 4 octants (z uses rfft so ±z is handled automatically)
        out_ft[:, :, :mx, :my, :mz] = self.compl_mul3d(
            x_ft[:, :, :mx, :my, :mz],
            self.weight1[:, :, :mx, :my, :mz],
        )
        out_ft[:, :, -mx:, :my, :mz] = self.compl_mul3d(
            x_ft[:, :, -mx:, :my, :mz],
            self.weight2[:, :, :mx, :my, :mz],
        )
        out_ft[:, :, :mx, -my:, :mz] = self.compl_mul3d(
            x_ft[:, :, :mx, -my:, :mz],
            self.weight3[:, :, :mx, :my, :mz],
        )
        out_ft[:, :, -mx:, -my:, :mz] = self.compl_mul3d(
            x_ft[:, :, -mx:, -my:, :mz],
            self.weight4[:, :, :mx, :my, :mz],
        )

        # Inverse FFT
        x_out = torch.fft.irfftn(out_ft, s=(Nx, Ny, Nz), dim=(-3, -2, -1))
        return x_out


# ── FNO3d ─────────────────────────────────────────────────────────────────────

class FNO3d(nn.Module):
    """
    3D Fourier Neural Operator.

    Expects input of shape:
      (B, C_in, Nx, Ny, Nz)

    Outputs:
      (B, C_out, Nx, Ny, Nz)
    """

    def __init__(
        self,
        modes_x=8,
        modes_y=8,
        modes_z=8,
        width=32,
        in_channels=7,
        out_channels=4,
        n_layers=4,
    ):
        super().__init__()

        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        self.width = width

        # Lifting layer: (C_in -> width)
        self.fc0 = nn.Conv3d(in_channels, width, kernel_size=1)

        self.convs = nn.ModuleList()
        self.ws = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                SpectralConv3d(width, width, modes_x, modes_y, modes_z)
            )
            self.ws.append(
                nn.Conv3d(width, width, kernel_size=1)
            )

        self.act = nn.GELU()

        # Projection layers: (width -> width -> out_channels)
        self.fc1 = nn.Conv3d(width, width, kernel_size=1)
        self.fc2 = nn.Conv3d(width, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x: (B, C_in, Nx, Ny, Nz)
        """
        x = self.fc0(x)  # (B, width, Nx, Ny, Nz)

        for conv, w in zip(self.convs, self.ws):
            x1 = conv(x)
            x2 = w(x)
            x = self.act(x1 + x2)

        x = self.act(self.fc1(x))
        x = self.fc2(x)   # (B, C_out, Nx, Ny, Nz)
        return x
