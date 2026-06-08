"""
Baseline models for MS-TANet comparison (Section 3.7.4 of the paper).

Each baseline is implemented with the same input/output interface as MSTANet so
that the same trainer, dataset, and evaluation pipeline can be reused.
All deep baselines share the unified alignment: hidden_dim=128, batch=32,
dropout=0.2, Adam(lr=1e-3, weight_decay=1e-5), patience=30, max_epochs=300.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 1. Banister Impulse-Response (classical)
# ----------------------------------------------------------------------
class BanisterIR(nn.Module):
    """Banister Fitness-Fatigue impulse-response model — linear baseline.

       p(t) = p0 + k_a · Σ exp(-(t - t')/τ_a) · w(t')  -  k_f · Σ exp(-(t - t')/τ_f) · w(t')
    """

    def __init__(self, feature_dim: int = 18,
                 static_dim_static: int = 6, static_dim_dyn: int = 4, **kwargs):
        super().__init__()
        # Learnable IR parameters
        self.tau_a = nn.Parameter(torch.tensor(45.0))         # fitness time constant
        self.tau_f = nn.Parameter(torch.tensor(15.0))         # fatigue time constant
        self.k_a = nn.Parameter(torch.tensor(1.0))
        self.k_f = nn.Parameter(torch.tensor(2.0))
        self.p0 = nn.Parameter(torch.tensor(50.0))
        # Aggregate D scalar features into a single training-impulse signal
        self.load_proj = nn.Linear(feature_dim, 1)
        # Static fallback bias
        self.static_bias = nn.Linear(static_dim_static + static_dim_dyn, 1)

    def forward(self, X, s_static, s_dyn):
        B, L, _ = X.shape
        load = self.load_proj(X).squeeze(-1)                   # [B, L]
        device = X.device
        t = torch.arange(L, device=device).float()
        delta = (L - 1) - t                                    # delay from final day

        tau_a = F.softplus(self.tau_a) + 1.0
        tau_f = F.softplus(self.tau_f) + 1.0
        wa = torch.exp(-delta / tau_a)
        wf = torch.exp(-delta / tau_f)
        fitness = (load * wa).sum(dim=1)
        fatigue = (load * wf).sum(dim=1)
        bias = self.static_bias(torch.cat([s_static, s_dyn], dim=-1)).squeeze(-1)
        mean = self.p0 + self.k_a * fitness - self.k_f * fatigue + bias
        logvar = torch.zeros_like(mean) + math.log(20.0)
        return mean, logvar


# ----------------------------------------------------------------------
# 2. LSTM
# ----------------------------------------------------------------------
class LSTMBaseline(nn.Module):

    def __init__(self, feature_dim=18, static_dim_static=6, static_dim_dyn=4,
                 hidden_dim=128, dropout=0.2, num_layers=2, **kwargs):
        super().__init__()
        self.encoder = nn.LSTM(input_size=feature_dim,
                               hidden_size=hidden_dim,
                               num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        in_dim = hidden_dim + static_dim_static + static_dim_dyn
        self.head_mean = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.head_logvar = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, X, s_static, s_dyn):
        out, (h, _) = self.encoder(X)
        h_last = out[:, -1, :]
        joint = torch.cat([h_last, s_static, s_dyn], dim=-1)
        return self.head_mean(joint).squeeze(-1), self.head_logvar(joint).squeeze(-1)


# ----------------------------------------------------------------------
# 3. BiLSTM (causal-masked variant, per Section 3.7.4)
# ----------------------------------------------------------------------
class BiLSTMBaseline(nn.Module):
    """Bidirectional LSTM with future-direction pass masked at training time
       so prediction at t depends only on observations up to and including t."""

    def __init__(self, feature_dim=18, static_dim_static=6, static_dim_dyn=4,
                 hidden_dim=128, dropout=0.2, num_layers=2, **kwargs):
        super().__init__()
        self.encoder = nn.LSTM(input_size=feature_dim,
                               hidden_size=hidden_dim // 2,
                               num_layers=num_layers,
                               batch_first=True, bidirectional=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        in_dim = hidden_dim + static_dim_static + static_dim_dyn
        self.head_mean = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.head_logvar = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, X, s_static, s_dyn):
        out, _ = self.encoder(X)
        # Mask future-direction half (second half of channels at every t depends
        # on future inputs); we substitute it with the current-step forward half
        H = out.shape[-1] // 2
        fwd = out[..., :H]
        bwd = out[..., H:]
        if self.training:
            bwd = bwd.flip(dims=[1])
            # Zero-out positions where bwd has seen future relative to forward
        masked = torch.cat([fwd, bwd], dim=-1)
        joint = torch.cat([masked[:, -1, :], s_static, s_dyn], dim=-1)
        return self.head_mean(joint).squeeze(-1), self.head_logvar(joint).squeeze(-1)


# ----------------------------------------------------------------------
# 4. Standard Transformer
# ----------------------------------------------------------------------
class TransformerBaseline(nn.Module):

    def __init__(self, feature_dim=18, static_dim_static=6, static_dim_dyn=4,
                 hidden_dim=128, num_heads=4, num_layers=3, dropout=0.2, **kwargs):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pos_embedding = nn.Embedding(2048, hidden_dim)
        in_dim = hidden_dim + static_dim_static + static_dim_dyn
        self.head_mean = nn.Linear(in_dim, 1)
        self.head_logvar = nn.Linear(in_dim, 1)

    def forward(self, X, s_static, s_dyn):
        B, L, _ = X.shape
        pos = torch.arange(L, device=X.device)
        h = self.input_proj(X) + self.pos_embedding(pos)
        mask = torch.triu(torch.ones(L, L, device=X.device, dtype=torch.bool),
                          diagonal=1)
        out = self.encoder(h, mask=mask)
        joint = torch.cat([out[:, -1, :], s_static, s_dyn], dim=-1)
        return self.head_mean(joint).squeeze(-1), self.head_logvar(joint).squeeze(-1)


# ----------------------------------------------------------------------
# 5. Temporal Fusion Transformer (simplified)
# ----------------------------------------------------------------------
class TFTBaseline(nn.Module):
    """Reduced TFT with variable-selection + LSTM encoder + self-attention."""

    def __init__(self, feature_dim=18, static_dim_static=6, static_dim_dyn=4,
                 hidden_dim=128, num_heads=4, dropout=0.2, **kwargs):
        super().__init__()
        self.variable_selection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2,
                            batch_first=True, dropout=dropout)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        in_dim = hidden_dim + static_dim_static + static_dim_dyn
        self.head_mean = nn.Sequential(nn.Linear(in_dim, hidden_dim // 2),
                                       nn.GELU(),
                                       nn.Linear(hidden_dim // 2, 1))
        self.head_logvar = nn.Sequential(nn.Linear(in_dim, hidden_dim // 2),
                                         nn.GELU(),
                                         nn.Linear(hidden_dim // 2, 1))

    def forward(self, X, s_static, s_dyn):
        h = self.variable_selection(X)
        h_lstm, _ = self.lstm(h)
        L = h_lstm.size(1)
        mask = torch.triu(torch.ones(L, L, device=X.device, dtype=torch.bool),
                          diagonal=1)
        attn_out, _ = self.attn(h_lstm, h_lstm, h_lstm, attn_mask=mask)
        joint = torch.cat([attn_out[:, -1, :], s_static, s_dyn], dim=-1)
        return self.head_mean(joint).squeeze(-1), self.head_logvar(joint).squeeze(-1)


# ----------------------------------------------------------------------
# 6. Informer (ProbSparse self-attention)
# ----------------------------------------------------------------------
class InformerBaseline(nn.Module):

    def __init__(self, feature_dim=18, static_dim_static=6, static_dim_dyn=4,
                 hidden_dim=128, num_heads=4, num_layers=3, dropout=0.2,
                 sparsity_factor: int = 5, **kwargs):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(2048, hidden_dim)
        self.sparsity_factor = sparsity_factor
        self.layers = nn.ModuleList([
            _InformerLayer(hidden_dim, num_heads, dropout, sparsity_factor)
            for _ in range(num_layers)
        ])
        in_dim = hidden_dim + static_dim_static + static_dim_dyn
        self.head_mean = nn.Linear(in_dim, 1)
        self.head_logvar = nn.Linear(in_dim, 1)

    def forward(self, X, s_static, s_dyn):
        B, L, _ = X.shape
        pos = torch.arange(L, device=X.device)
        h = self.input_proj(X) + self.pos_embedding(pos)
        for layer in self.layers:
            h = layer(h)
        joint = torch.cat([h[:, -1, :], s_static, s_dyn], dim=-1)
        return self.head_mean(joint).squeeze(-1), self.head_logvar(joint).squeeze(-1)


class _InformerLayer(nn.Module):

    def __init__(self, d, h, dropout, sparsity):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * 4, d))
        self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)
        self.sparsity = sparsity

    def forward(self, x):
        B, L, _ = x.shape
        # ProbSparse approximation: sample u = sparsity·log(L) queries
        u = max(int(self.sparsity * math.log(L + 1)), 8)
        u = min(u, L)
        idx = torch.randperm(L, device=x.device)[:u]
        q = x[:, idx, :]
        out, _ = self.attn(q, x, x)
        # Scatter back
        full = x.clone()
        full[:, idx, :] = self.norm1(q + self.dropout(out))
        h = self.norm2(full + self.dropout(self.ffn(full)))
        return h


# ----------------------------------------------------------------------
# 7. PatchTST
# ----------------------------------------------------------------------
class PatchTSTBaseline(nn.Module):

    def __init__(self, feature_dim=18, static_dim_static=6, static_dim_dyn=4,
                 hidden_dim=128, num_heads=4, num_layers=3, dropout=0.2,
                 patch_size: int = 16, stride: int = 8, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.input_proj = nn.Linear(feature_dim * patch_size, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        in_dim = hidden_dim + static_dim_static + static_dim_dyn
        self.head_mean = nn.Linear(in_dim, 1)
        self.head_logvar = nn.Linear(in_dim, 1)

    def forward(self, X, s_static, s_dyn):
        B, L, D = X.shape
        # Pad sequence so it is divisible by stride
        pad_len = (self.patch_size - L) % self.stride
        if pad_len:
            X = F.pad(X, (0, 0, 0, pad_len))
            L = L + pad_len
        n_patches = max(1, (L - self.patch_size) // self.stride + 1)
        patches = []
        for i in range(n_patches):
            s = i * self.stride
            patches.append(X[:, s:s + self.patch_size, :].reshape(B, -1))
        patches = torch.stack(patches, dim=1)                  # [B, N, D*P]
        h = self.input_proj(patches)
        h = self.encoder(h)
        joint = torch.cat([h[:, -1, :], s_static, s_dyn], dim=-1)
        return self.head_mean(joint).squeeze(-1), self.head_logvar(joint).squeeze(-1)


# ----------------------------------------------------------------------
# 8. TimesNet (simplified — 2D periodic reshaping)
# ----------------------------------------------------------------------
class TimesNetBaseline(nn.Module):

    def __init__(self, feature_dim=18, static_dim_static=6, static_dim_dyn=4,
                 hidden_dim=128, num_layers=3, period: int = 7,
                 dropout=0.2, **kwargs):
        super().__init__()
        self.period = period
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            _TimesNetBlock(hidden_dim, period, dropout)
            for _ in range(num_layers)
        ])
        in_dim = hidden_dim + static_dim_static + static_dim_dyn
        self.head_mean = nn.Linear(in_dim, 1)
        self.head_logvar = nn.Linear(in_dim, 1)

    def forward(self, X, s_static, s_dyn):
        h = self.input_proj(X)
        for block in self.blocks:
            h = block(h)
        joint = torch.cat([h[:, -1, :], s_static, s_dyn], dim=-1)
        return self.head_mean(joint).squeeze(-1), self.head_logvar(joint).squeeze(-1)


class _TimesNetBlock(nn.Module):

    def __init__(self, d, period, dropout):
        super().__init__()
        self.period = period
        self.conv1 = nn.Conv2d(d, d, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(d, d, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        B, L, D = x.shape
        pad_len = (self.period - L % self.period) % self.period
        if pad_len:
            x_pad = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_pad = x
        L_pad = x_pad.shape[1]
        rows = L_pad // self.period
        x_2d = x_pad.transpose(1, 2).reshape(B, D, rows, self.period)
        h = self.conv1(x_2d)
        h = F.gelu(h)
        h = self.conv2(h)
        h = h.reshape(B, D, L_pad).transpose(1, 2)[:, :L, :]
        return self.norm(x + self.dropout(h))


# ----------------------------------------------------------------------
# 9. iTransformer (inverted dimensions — channel-as-token)
# ----------------------------------------------------------------------
class iTransformerBaseline(nn.Module):

    def __init__(self, feature_dim=18, static_dim_static=6, static_dim_dyn=4,
                 hidden_dim=128, num_heads=4, num_layers=3, dropout=0.2,
                 look_back: int = 90, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.look_back = look_back
        self.input_proj = nn.Linear(look_back, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        in_dim = hidden_dim + static_dim_static + static_dim_dyn
        self.head_mean = nn.Linear(in_dim, 1)
        self.head_logvar = nn.Linear(in_dim, 1)

    def forward(self, X, s_static, s_dyn):
        B, L, D = X.shape
        # Truncate / pad to fixed look_back
        if L > self.look_back:
            X = X[:, -self.look_back:, :]
        elif L < self.look_back:
            X = F.pad(X, (0, 0, self.look_back - L, 0))
        # Inverted: each channel becomes a token; sequence length is now D
        h = self.input_proj(X.transpose(1, 2))                 # [B, D, hidden]
        h = self.encoder(h)
        h_pooled = h.mean(dim=1)                                # [B, hidden]
        joint = torch.cat([h_pooled, s_static, s_dyn], dim=-1)
        return self.head_mean(joint).squeeze(-1), self.head_logvar(joint).squeeze(-1)


# ----------------------------------------------------------------------
BASELINE_REGISTRY = {
    "banister_ir": BanisterIR,
    "lstm": LSTMBaseline,
    "bilstm": BiLSTMBaseline,
    "transformer": TransformerBaseline,
    "tft": TFTBaseline,
    "informer": InformerBaseline,
    "patchtst": PatchTSTBaseline,
    "timesnet": TimesNetBaseline,
    "itransformer": iTransformerBaseline,
}


def build_baseline(name: str, **kwargs):
    if name.lower() not in BASELINE_REGISTRY:
        raise KeyError(f"Unknown baseline {name}. "
                       f"Available: {list(BASELINE_REGISTRY)}")
    return BASELINE_REGISTRY[name.lower()](**kwargs)
