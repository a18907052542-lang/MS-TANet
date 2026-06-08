"""
Physics-Constrained Regularization Loss.

Implements Eq. (8) - (12) of the paper:

    L_total = L_pred + α · L_phys                                                          (8)
    L_pred  = (1/NT) Σ_{i,t} ρ_δ( y_i(t) - ŷ_i(t) )                                        (9)
    L_phys  = L_decay + L_bound                                                           (10)
    L_decay = ReLU(-λ) + (1/|S|) Σ_{(t,t',t'') ∈ S} ReLU(a_{t,t''} - a_{t,t'})            (11)
    L_bound = (1/NT) Σ_{i,t} ReLU(|ŷ_i(t) - ŷ_i(t-1)| - Δ_max)                            (12)

Heteroscedastic NLL is added when deep ensembling is used (Eq. 13):

    L_NLL = (1/(2NT)) Σ_{i,t} [(y - ŷ)^2 / σ^2 + log σ^2]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsConstrainedLoss(nn.Module):

    def __init__(self,
                 huber_delta: float = 1.0,
                 delta_max: float = 6.0,
                 alpha: float = 0.5,
                 use_nll: bool = True,
                 nll_weight: float = 0.3,
                 triplet_samples: int = 1024):
        super().__init__()
        self.huber_delta = huber_delta
        self.delta_max = delta_max
        self.alpha = alpha
        self.use_nll = use_nll
        self.nll_weight = nll_weight
        self.triplet_samples = triplet_samples

    # ------------------------------------------------------------------
    def huber(self, residual: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(residual, torch.zeros_like(residual),
                            delta=self.huber_delta, reduction="mean")

    # ------------------------------------------------------------------
    def decay_constraint(self,
                         decay_lambda: torch.Tensor,
                         attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Eq. (11): positivity + monotonicity on attention weights.

        decay_lambda : [num_heads] (already-softplus'd, so non-negative; we still
                       penalise residual negativity for the raw parameter form).
        attention_weights : [batch, L, L]
        """
        # Penalise any negative entries (raw_lambda is unconstrained so softplus
        # guarantees positivity, but we keep the term for completeness).
        lam_term = F.relu(-decay_lambda).mean()

        batch, L, _ = attention_weights.shape
        device = attention_weights.device

        # Random triplet sampling: t, t', t'' with (t - t') < (t - t'')
        # i.e. t' is closer to t than t''
        n = min(self.triplet_samples, batch * 8)
        if L < 3 or n == 0:
            return lam_term

        idx_t = torch.randint(2, L, (n,), device=device)
        # delta1 < delta2 — t' closer than t''
        delta1 = torch.randint(1, L, (n,), device=device)
        delta1 = torch.minimum(delta1, idx_t)
        delta2 = torch.randint(1, L, (n,), device=device)
        delta2 = torch.minimum(delta2, idx_t)
        swap = (delta1 > delta2)
        d1 = torch.where(swap, delta2, delta1)
        d2 = torch.where(swap, delta1, delta2)
        valid = (d1 < d2)
        idx_t = idx_t[valid]
        d1 = d1[valid]
        d2 = d2[valid]
        if idx_t.numel() == 0:
            return lam_term

        idx_tp = idx_t - d1
        idx_tpp = idx_t - d2
        b_idx = torch.randint(0, batch, (idx_t.numel(),), device=device)

        a_tpp = attention_weights[b_idx, idx_t, idx_tpp]
        a_tp = attention_weights[b_idx, idx_t, idx_tp]
        mono = F.relu(a_tpp - a_tp).mean()

        return lam_term + mono

    # ------------------------------------------------------------------
    def bound_constraint(self, predictions_per_athlete) -> torch.Tensor:
        """
        Eq. (12): adaptation boundedness.

        predictions_per_athlete : dict {athlete_id : Tensor[seq_len]}
                                 OR Tensor[batch] of consecutive-day predictions.
        """
        if predictions_per_athlete is None:
            return torch.tensor(0.0)
        if isinstance(predictions_per_athlete, torch.Tensor):
            preds = predictions_per_athlete
            if preds.dim() == 1 or preds.shape[-1] < 2:
                return torch.tensor(0.0, device=preds.device)
            diff = torch.abs(preds[..., 1:] - preds[..., :-1])
            return F.relu(diff - self.delta_max).mean()

        terms = []
        for seq in predictions_per_athlete.values():
            if seq.numel() < 2:
                continue
            diff = torch.abs(seq[1:] - seq[:-1])
            terms.append(F.relu(diff - self.delta_max).mean())
        if not terms:
            return torch.tensor(0.0)
        return torch.stack(terms).mean()

    # ------------------------------------------------------------------
    def forward(self,
                pred_mean: torch.Tensor,
                pred_logvar: torch.Tensor,
                target: torch.Tensor,
                decay_lambda: torch.Tensor = None,
                attention_weights: torch.Tensor = None,
                prediction_sequences=None,
                phys_active: bool = True) -> dict:
        """
        Returns a dict with the breakdown of all loss components.
        """
        residual = target - pred_mean
        L_pred = self.huber(residual)

        loss_dict = {"L_pred": L_pred}
        total = L_pred

        if self.use_nll:
            var = torch.exp(pred_logvar).clamp_min(1e-4)
            L_nll = 0.5 * (residual.pow(2) / var + pred_logvar).mean()
            loss_dict["L_nll"] = L_nll
            total = total + self.nll_weight * L_nll

        if phys_active:
            if attention_weights is not None and decay_lambda is not None:
                L_decay = self.decay_constraint(decay_lambda, attention_weights)
            else:
                L_decay = torch.tensor(0.0, device=pred_mean.device)

            L_bound = self.bound_constraint(prediction_sequences)
            L_phys = L_decay + L_bound
            loss_dict["L_decay"] = L_decay
            loss_dict["L_bound"] = L_bound
            loss_dict["L_phys"] = L_phys
            total = total + self.alpha * L_phys

        loss_dict["L_total"] = total
        return loss_dict


def heteroscedastic_nll(pred_mean: torch.Tensor,
                        pred_logvar: torch.Tensor,
                        target: torch.Tensor) -> torch.Tensor:
    """Eq. (13)."""
    residual = target - pred_mean
    var = torch.exp(pred_logvar).clamp_min(1e-4)
    return 0.5 * (residual.pow(2) / var + pred_logvar).mean()
