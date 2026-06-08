"""
Full MS-TANet model — assembles the three core modules in the
"local → long-range → individual" three-tier complementary structure.

Computational flow (Section 3.1, Figure 2):

    X_i ∈ ℝ^{L × D}
       │
       ▼  MS-DCC (Eq. 2-3)
    H_fused ∈ ℝ^{L × D_h}
       │
       ▼  SD-SA (Eq. 4-5)
    H_attn ∈ ℝ^{L × D_h}
       │
       ▼  IE-FM (Eq. 6-7)
    H_i ∈ ℝ^{L × D_h}
       │
       ▼  Prediction head + heteroscedastic NLL
    ŷ_i(t), log σ^2_i(t)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ms_tanet.ms_dcc import MultiScaleDilatedCausalConv
from ms_tanet.sd_sa import SparseDecaySelfAttention
from ms_tanet.ie_fm import IndividualEmbeddingFeatureModulation


class MSTANet(nn.Module):
    """Multi-Scale Temporal Attention Network."""

    def __init__(self,
                 feature_dim: int = 18,
                 static_dim_static: int = 6,
                 static_dim_dyn: int = 4,
                 hidden_dim: int = 128,
                 embedding_dim: int = 32,
                 dilation_rates=(1, 7, 28),
                 conv_kernel_size: int = 3,
                 num_attention_heads: int = 4,
                 top_k: int = 20,
                 lsh_hyperplanes: int = 4,
                 initial_lambda: float = 0.04,
                 dropout: float = 0.2,
                 # Ablation switches
                 use_multi_scale: bool = True,
                 use_sparse_decay_attention: bool = True,
                 use_individual_embedding: bool = True,
                 widened_single_scale_channels: int = None):
        super().__init__()

        # Ablation: drop multi-scale and use a single dilation
        if not use_multi_scale:
            if widened_single_scale_channels is not None:
                effective_dim = widened_single_scale_channels
                # Single-branch model with widened channels for parameter-matching
                self.ms_dcc = MultiScaleDilatedCausalConv(
                    feature_dim, effective_dim,
                    kernel_size=conv_kernel_size,
                    dilation_rates=(1,),
                    dropout=dropout)
                self.pre_attn_proj = nn.Linear(effective_dim, hidden_dim) \
                    if effective_dim != hidden_dim else nn.Identity()
            else:
                self.ms_dcc = MultiScaleDilatedCausalConv(
                    feature_dim, hidden_dim,
                    kernel_size=conv_kernel_size,
                    dilation_rates=(1,),
                    dropout=dropout)
                self.pre_attn_proj = nn.Identity()
        else:
            self.ms_dcc = MultiScaleDilatedCausalConv(
                feature_dim, hidden_dim,
                kernel_size=conv_kernel_size,
                dilation_rates=dilation_rates,
                dropout=dropout)
            self.pre_attn_proj = nn.Identity()

        # Attention
        self.use_sparse_decay_attention = use_sparse_decay_attention
        if use_sparse_decay_attention:
            self.attn = SparseDecaySelfAttention(
                hidden_dim=hidden_dim,
                num_heads=num_attention_heads,
                top_k=top_k,
                lsh_hyperplanes=lsh_hyperplanes,
                initial_lambda=initial_lambda,
                dropout=dropout,
            )
        else:
            # Standard self-attention (no decay, no sparsity)
            self.attn = StandardMultiHeadAttention(
                hidden_dim=hidden_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
            )

        # Individual embedding
        self.ie_fm = IndividualEmbeddingFeatureModulation(
            static_dim_static=static_dim_static,
            static_dim_dyn=static_dim_dyn,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_modulation=use_individual_embedding,
        )

        # Prediction head — mean + log-variance (for heteroscedastic NLL)
        self.pool = TemporalPooling(hidden_dim, dropout=dropout)
        self.head_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.head_logvar = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Cached values for analysis
        self._last_attn_weights = None
        self._last_gates = None
        self._last_gamma = None
        self._last_beta = None

    # ------------------------------------------------------------------
    def forward(self,
                X: torch.Tensor,
                s_static: torch.Tensor,
                s_dyn: torch.Tensor,
                return_extras: bool = False):
        """
        X        : [batch, L, D]
        s_static : [batch, D_s_static]
        s_dyn    : [batch, D_s_dyn]

        returns:
            mean    : [batch]
            logvar  : [batch]
            extras  : dict (only if return_extras)
        """
        H_fused, gates, branch_outputs = self.ms_dcc(X)
        H_fused = self.pre_attn_proj(H_fused)

        H_attn, attn_weights = self.attn(H_fused)
        H_i, gamma, beta, embedding = self.ie_fm(H_attn, s_static, s_dyn)

        pooled = self.pool(H_i)
        mean = self.head_mean(pooled).squeeze(-1)
        logvar = self.head_logvar(pooled).squeeze(-1)

        self._last_attn_weights = attn_weights
        self._last_gates = gates
        self._last_gamma = gamma
        self._last_beta = beta

        if return_extras:
            return mean, logvar, {
                "attention_weights": attn_weights,
                "gating_weights": gates,
                "gamma": gamma,
                "beta": beta,
                "embedding": embedding,
                "branch_outputs": branch_outputs,
            }
        return mean, logvar

    # ------------------------------------------------------------------
    @property
    def decay_lambda(self):
        if hasattr(self.attn, "decay_lambda"):
            return self.attn.decay_lambda
        return None

    @property
    def half_life_days(self):
        if hasattr(self.attn, "half_life_days"):
            return self.attn.half_life_days
        return None

    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TemporalPooling(nn.Module):
    """Attention-pooled temporal summary that emphasises recent days."""

    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [batch, L, D_h]
        scores = (self.proj(x) * self.query).sum(dim=-1)      # [batch, L]
        weights = F.softmax(scores, dim=-1)                   # [batch, L]
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)       # [batch, D_h]
        pooled = self.dropout(pooled)
        return self.norm(pooled)


class StandardMultiHeadAttention(nn.Module):
    """Vanilla causal multi-head self-attention used in ablation."""

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor):
        L = x.size(1)
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool),
                          diagonal=1)
        out, w = self.attn(x, x, x, attn_mask=mask, need_weights=True,
                           average_attn_weights=True)
        return self.norm(out + x), w.detach()
