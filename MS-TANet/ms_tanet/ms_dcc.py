"""
Multi-Scale Dilated Causal Convolution (MS-DCC) module.
Implements Eq. (2) and Eq. (3) of the paper:

    H_m(t) = σ( Σ_{k=0}^{K-1} W_m^{(l,k)} · H(t - d_m · k) + b_m^{(l)} )                  (2)

    H_fused(t) = Σ_{m=1}^{3} g_m(t) ⊙ H_m(t),
        g(t) = Softmax(W_g [H_1(t); H_2(t); H_3(t)])                                       (3)

Three parallel branches with dilation rates d_1=1, d_2=7, d_3=28 produce
representations at the daily, weekly, and monthly receptive fields.  The
learnable gating fuses them adaptively at every time step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDilatedConv1d(nn.Module):
    """1-D causal convolution with a given dilation rate."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        # Left-pad so causality is preserved.
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time]
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class MultiScaleDilatedCausalConv(nn.Module):
    """Three-branch dilated causal convolution module with learnable gating fusion."""

    def __init__(self, in_dim: int, hidden_dim: int,
                 kernel_size: int = 3,
                 dilation_rates=(1, 7, 28),
                 dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dilation_rates = tuple(dilation_rates)
        self.num_branches = len(self.dilation_rates)

        # Three parallel branches
        self.branches = nn.ModuleList()
        for d in self.dilation_rates:
            branch = nn.Sequential(
                CausalDilatedConv1d(in_dim, hidden_dim, kernel_size, d),
                nn.GELU(),
                CausalDilatedConv1d(hidden_dim, hidden_dim, kernel_size, d),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.branches.append(branch)

        # Gating: W_g  ∈ ℝ^{3 × 3·D_h}
        gating_in = self.num_branches * hidden_dim
        self.gating_proj = nn.Linear(gating_in, self.num_branches)

        # Residual projection so the module behaves cleanly when input dim != hidden_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim) \
            if in_dim != hidden_dim else nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor):
        """
        x : [batch, L, D_in]
        returns:
            H_fused : [batch, L, D_h]
            gates   : [batch, L, num_branches]   (g_m(t) for analysis)
            H_branches : list of [batch, L, D_h]  (per-branch features)
        """
        # Conv1d expects [batch, channels, time]
        x_t = x.transpose(1, 2)  # [batch, D_in, L]

        branch_outputs = []
        for branch in self.branches:
            h = branch(x_t)            # [batch, D_h, L]
            branch_outputs.append(h.transpose(1, 2))  # [batch, L, D_h]

        # Stack and gate
        concat = torch.cat(branch_outputs, dim=-1)        # [batch, L, num*D_h]
        gates = F.softmax(self.gating_proj(concat), dim=-1)  # [batch, L, num]

        H_fused = torch.zeros_like(branch_outputs[0])
        for m, h_m in enumerate(branch_outputs):
            H_fused = H_fused + gates[..., m:m+1] * h_m

        H_fused = self.layer_norm(H_fused + self.input_proj(x))
        return H_fused, gates, branch_outputs

    def freeze_low_level(self):
        """Stage-2 of the curriculum freezes low-level conv weights."""
        for branch in self.branches:
            # Freeze the first conv layer of each branch.
            for module in list(branch.children())[:2]:
                for p in module.parameters():
                    p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
