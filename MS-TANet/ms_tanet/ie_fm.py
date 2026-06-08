"""
Individual Embedding and Feature Modulation (IE-FM) module.

Implements Eq. (6) and Eq. (7) of the paper:

    e_i  = ReLU(W_s · s_i)
    γ_i  = σ(W_γ · e_i)
    β_i  = W_β · e_i                                                                       (6)

    H_i(t) = γ_i ⊙ H_fused(t) + β_i                                                        (7)

Two-tier conditioning strategy (Section 3.4):

  Tier 1 (strictly static): sport category, primary position/event, sex, age,
                            dominant side, years of training.  Encoded once per
                            season and frozen.

  Tier 2 (quasi-dynamic):   body mass, body-fat %, VO2max, resting HR.
                            Refreshed monthly; held constant within each month
                            and broadcast across daily time steps.

Concatenation s_i(t) = [s_i^static ; s_i^dyn(t)] is shared through a single
encoder W_s, so the modulation factors (γ_i, β_i) adapt to slow physiological
drift while keeping invariant attributes stable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IndividualEmbeddingFeatureModulation(nn.Module):

    def __init__(self,
                 static_dim_static: int,
                 static_dim_dyn: int,
                 embedding_dim: int = 32,
                 hidden_dim: int = 128,
                 dropout: float = 0.2,
                 use_modulation: bool = True):
        super().__init__()
        self.static_dim_static = static_dim_static
        self.static_dim_dyn = static_dim_dyn
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.use_modulation = use_modulation

        # Shared two-layer fully connected encoder W_s
        in_dim = static_dim_static + static_dim_dyn
        self.static_encoder = nn.Sequential(
            nn.Linear(in_dim, embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(inplace=True),
        )

        # γ_i (sigmoid-bounded scaling) and β_i (unconstrained offset)
        self.scale_proj = nn.Linear(embedding_dim, hidden_dim)
        self.offset_proj = nn.Linear(embedding_dim, hidden_dim)
        # Initialise so that γ ≈ 1, β ≈ 0 at training start
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.offset_proj.weight)
        nn.init.zeros_(self.offset_proj.bias)

    def forward(self,
                features_temporal: torch.Tensor,
                s_static: torch.Tensor,
                s_dyn: torch.Tensor):
        """
        features_temporal : [batch, L, D_h]   (output of MS-DCC / SD-SA)
        s_static          : [batch, D_s_static]
        s_dyn             : [batch, D_s_dyn]  (monthly snapshot)

        returns:
            modulated_features : [batch, L, D_h]
            gamma, beta        : [batch, D_h]
            embedding          : [batch, D_e]
        """
        if not self.use_modulation:
            # Knockout variant — bypass modulation but still build embedding for the head
            zeros = torch.zeros(features_temporal.size(0), self.embedding_dim,
                                device=features_temporal.device,
                                dtype=features_temporal.dtype)
            return features_temporal, None, None, zeros

        s = torch.cat([s_static, s_dyn], dim=-1)             # [batch, D_s]
        e_i = self.static_encoder(s)                          # [batch, D_e]

        gamma_raw = self.scale_proj(e_i)
        gamma = torch.sigmoid(gamma_raw) + 0.5                # range (0.5, 1.5)
        beta = self.offset_proj(e_i)                          # [batch, D_h]

        # Broadcast across time
        modulated = gamma.unsqueeze(1) * features_temporal + beta.unsqueeze(1)
        return modulated, gamma, beta, e_i
