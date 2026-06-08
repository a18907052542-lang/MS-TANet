"""
Sparse Decay Self-Attention (SD-SA) module.

Implements Eq. (4) and Eq. (5) of the paper:

    a_{t,t'} = Softmax( (Q_t K_{t'}^T / sqrt(D_k)) - λ · (t - t') )                       (4)
    SD-SA(Q,K,V) = Σ_{t' ∈ N_k(t)} a_{t,t'} · V_{t'}                                       (5)

Key features:
  * Learnable decay rate λ > 0 — exponential decay bias on attention logits.
  * Top-k sparsification (k = 20) — only k key positions per query.
  * Locality-Sensitive Hashing (LSH) pre-screening with H = 4 hyperplanes —
    candidate-key set |C_t| ≈ 3·k = 60 — reducing the asymptotic complexity
    from O(L^2 · D_k) to O(L · (log L + k · D_k)).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseDecaySelfAttention(nn.Module):
    """Multi-head SD-SA with learnable exponential decay and LSH-screened top-k."""

    def __init__(self,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 top_k: int = 20,
                 lsh_hyperplanes: int = 4,
                 initial_lambda: float = 0.04,
                 use_lsh: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.top_k = top_k
        self.lsh_hyperplanes = lsh_hyperplanes
        self.use_lsh = use_lsh

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Learnable decay rate (one λ per head); raw_lambda is unconstrained,
        # the effective λ = softplus(raw_lambda) is guaranteed positive.
        init_raw = math.log(math.expm1(initial_lambda))
        self.raw_lambda = nn.Parameter(torch.full((num_heads,), init_raw))

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    @property
    def decay_lambda(self) -> torch.Tensor:
        """Effective λ (positive via softplus)."""
        return F.softplus(self.raw_lambda)

    @property
    def half_life_days(self) -> torch.Tensor:
        """t_{1/2} = ln 2 / λ — physiologically interpretable."""
        return math.log(2.0) / (self.decay_lambda + 1e-8)

    # ------------------------------------------------------------------
    # LSH pre-screening
    # ------------------------------------------------------------------
    def _lsh_candidate_set(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Build the candidate-key index set C_t for every query position t.

        q, k : [batch, num_heads, L, head_dim]
        returns candidate mask : [batch, num_heads, L_q, L_k]  (bool)
        """
        batch, H, L_q, D = q.shape
        L_k = k.shape[2]

        # Random hyperplanes — fixed within a forward pass, regenerated per call
        hyperplanes = torch.randn(self.lsh_hyperplanes, D,
                                  device=q.device, dtype=q.dtype)

        # Sign pattern of the projection
        q_sign = (q @ hyperplanes.t() > 0).int()   # [batch, H, L_q, num_hp]
        k_sign = (k @ hyperplanes.t() > 0).int()   # [batch, H, L_k, num_hp]

        # Two positions are in the same bucket iff at least one hyperplane sign matches
        # Compute equality across hyperplanes:
        eq = (q_sign.unsqueeze(3) == k_sign.unsqueeze(2))   # [batch, H, L_q, L_k, num_hp]
        candidate_mask = eq.any(dim=-1)                     # [batch, H, L_q, L_k]
        return candidate_mask

    def forward(self, x: torch.Tensor):
        """
        x : [batch, L, D_h]
        returns:
            output : [batch, L, D_h]
            attn_weights_mean : [batch, L, L] mean across heads (for visualisation)
        """
        batch, L, _ = x.shape

        # Project
        q = self.q_proj(x).view(batch, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, L, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v : [batch, num_heads, L, head_dim]

        # Causal mask — query t can only attend to keys t' <= t
        time_idx = torch.arange(L, device=x.device)
        delta_t = time_idx.unsqueeze(0) - time_idx.unsqueeze(1)            # [L, L], delta = t - t'
        causal = (delta_t >= 0)                                            # bool [L, L]

        # Compute scaled dot-product
        scale = math.sqrt(self.head_dim)
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / scale             # [batch, H, L_q, L_k]

        # Exponential decay bias — per head λ
        decay = self.decay_lambda.view(1, -1, 1, 1) * delta_t.unsqueeze(0).unsqueeze(0)
        scores = scores - decay.float()

        # Mask non-causal positions
        scores = scores.masked_fill(~causal, float("-inf"))

        # LSH candidate screening
        if self.use_lsh and L > 4 * self.top_k:
            candidate_mask = self._lsh_candidate_set(q, k)                 # [batch, H, L_q, L_k]
            candidate_mask = candidate_mask & causal.unsqueeze(0).unsqueeze(0)
            # If candidate set is too small for some query, fall back to top-k on full set
            cand_sum = candidate_mask.sum(dim=-1, keepdim=True)
            small = (cand_sum < self.top_k)
            # When small, allow attending to anything causal
            allow = candidate_mask | small
            scores = scores.masked_fill(~allow, float("-inf"))

        # Top-k sparsification per query
        k_eff = min(self.top_k, L)
        topk_vals, topk_idx = scores.topk(k_eff, dim=-1)
        topk_mask = torch.zeros_like(scores, dtype=torch.bool)
        topk_mask.scatter_(-1, topk_idx, True)
        scores = scores.masked_fill(~topk_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v)                     # [batch, H, L, D_head]
        out = out.transpose(1, 2).contiguous().view(batch, L, self.hidden_dim)
        out = self.out_proj(out)
        out = self.layer_norm(out + x)

        attn_mean = attn.mean(dim=1).detach()                              # [batch, L, L]
        return out, attn_mean
