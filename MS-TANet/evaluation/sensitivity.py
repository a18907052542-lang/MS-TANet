"""
Sensitivity analysis for the physics-constraint coefficient α — Section 4.8, Table 11.

Sweep α over {0, 0.1, 0.3, 0.5, 0.7, 1.0} on the longitudinal dataset under
5-fold blocked TS-CV, with all other hyperparameters held constant.
"""

import numpy as np
import torch
from typing import Dict

from ms_tanet.model import MSTANet
from training.curriculum import CurriculumSchedule
from evaluation.cross_validation import run_chronological_cv


def alpha_sensitivity_sweep(samples,
                            base_kwargs: dict,
                            alpha_values=(0.0, 0.1, 0.3, 0.5, 0.7, 1.0),
                            device: str = "cpu",
                            max_epochs: int = 30) -> Dict[float, Dict]:
    results = {}
    for alpha in alpha_values:
        torch.manual_seed(42); np.random.seed(42)
        curriculum = CurriculumSchedule(alpha_max=alpha)
        builder = lambda: MSTANet(**base_kwargs)
        res = run_chronological_cv(
            builder, samples,
            device=device, max_epochs=max_epochs,
            curriculum=curriculum, alpha_max=alpha,
        )

        # Auxiliary metrics: negative-λ rate, attention-monotonicity score
        neg_lambda_rate, monotone_score = _evaluate_interpretability(
            builder, samples, device=device, max_epochs=max_epochs,
            alpha_max=alpha, curriculum=curriculum,
        )
        results[alpha] = {
            "RMSE": res["RMSE"]["mean"],
            "MAE": res["MAE"]["mean"],
            "R2": res["R2"]["mean"],
            "NRMSE": res["NRMSE"]["mean"],
            "negative_lambda_rate_pct": neg_lambda_rate,
            "attention_monotonicity_score": monotone_score,
        }
    return results


def _evaluate_interpretability(builder, samples, device,
                                max_epochs, alpha_max, curriculum):
    """Estimate negative-λ rate and attention-monotonicity score."""
    model = builder().to(device)
    lam = model.decay_lambda
    if lam is None:
        return 0.0, 0.0
    neg_lambda_rate = 100.0 * float((lam < 0).float().mean().item())

    # Sample attention from a forward pass; compute fraction of pairs where
    # closer-in-time keys get larger attention than farther-in-time keys.
    if len(samples) == 0:
        return neg_lambda_rate, 0.0
    s = samples[0]
    X = torch.from_numpy(s["X"]).unsqueeze(0).to(device)
    ss = torch.from_numpy(s["s_static"]).unsqueeze(0).to(device)
    sd = torch.from_numpy(s["s_dyn"]).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        _ = model(X, ss, sd)
    attn = model._last_attn_weights
    if attn is None:
        return neg_lambda_rate, 0.0
    score = _monotonicity_score(attn[0].cpu().numpy())
    return neg_lambda_rate, score


def _monotonicity_score(attn_matrix: np.ndarray) -> float:
    """Fraction of (t, t', t'') triplets with t' closer to t than t''
       for which a_{t,t'} >= a_{t,t''}."""
    L = attn_matrix.shape[0]
    if L < 3:
        return 0.0
    n_total = 0
    n_satisfy = 0
    rng = np.random.default_rng(0)
    n_samples = min(4000, L * (L - 1) // 2)
    for _ in range(n_samples):
        t = rng.integers(2, L)
        d1 = rng.integers(1, t + 1)
        d2 = rng.integers(1, t + 1)
        if d1 >= d2:
            continue
        tp = t - d1
        tpp = t - d2
        n_total += 1
        if attn_matrix[t, tp] >= attn_matrix[t, tpp]:
            n_satisfy += 1
    return float(n_satisfy / max(1, n_total))
