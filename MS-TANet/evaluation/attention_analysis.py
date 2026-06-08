"""
Attention-mechanism robustness diagnostics — Section 4.4 of the paper.

Three falsification checks:
  1. Window removal:  zero-out the top-decay-weight pre-competition window of
                       width w in the input and observe ΔR² on test predictions.
  2. Perturbation:    add Gaussian noise N(0, 0.1) to the same window and
                       measure prediction-error increase.
  3. Randomization:   shuffle the top-decay-weight window randomly and measure
                       ΔR² collapse — should be the largest of the three.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List

from training.dataset import AthleteDataset


def _build_baseline_predictions(model, samples, device, window_features):
    loader = DataLoader(AthleteDataset(samples), batch_size=32, shuffle=False)
    model.eval()
    means, targets = [], []
    with torch.no_grad():
        for X, s_static, s_dyn, y, _ in loader:
            X = X.to(device); s_static = s_static.to(device); s_dyn = s_dyn.to(device)
            mean, _ = model(X, s_static, s_dyn)
            means.extend(mean.cpu().numpy().tolist())
            targets.extend(y.numpy().tolist())
    return np.array(means), np.array(targets)


def _select_window_mask(L: int, sport: str, window_width: int = 14):
    """Top-decay-weight window — pre-competition phase, last 14 days of stage 2."""
    end = L
    start = max(0, end - window_width)
    mask = np.zeros(L, dtype=bool)
    mask[start:end] = True
    return mask


def window_removal(model, samples, device,
                   window_width: int = 14) -> Dict:
    means_base, targets = _build_baseline_predictions(model, samples, device, None)
    r2_base = _r2(targets, means_base)

    perturbed_samples = []
    for s in samples:
        s2 = dict(s)
        X = s["X"].copy()
        mask = _select_window_mask(X.shape[0], s["sport"], window_width)
        X[mask] = 0.0
        s2["X"] = X
        perturbed_samples.append(s2)
    means_perturbed, _ = _build_baseline_predictions(model, perturbed_samples,
                                                      device, None)
    r2_perturbed = _r2(targets, means_perturbed)
    delta = r2_perturbed - r2_base
    return {"r2_baseline": r2_base, "r2_after": r2_perturbed,
            "delta_R2": delta, "window_width_days": window_width}


def perturbation(model, samples, device,
                 sigma: float = 0.1, window_width: int = 14) -> Dict:
    means_base, targets = _build_baseline_predictions(model, samples, device, None)
    err_base = float(np.mean(np.abs(targets - means_base)))

    rng = np.random.default_rng(0)
    perturbed_samples = []
    for s in samples:
        s2 = dict(s)
        X = s["X"].copy()
        mask = _select_window_mask(X.shape[0], s["sport"], window_width)
        X[mask] += rng.normal(0.0, sigma, size=X[mask].shape)
        s2["X"] = X
        perturbed_samples.append(s2)
    means_perturbed, _ = _build_baseline_predictions(model, perturbed_samples,
                                                      device, None)
    err_perturbed = float(np.mean(np.abs(targets - means_perturbed)))
    delta_err = err_perturbed - err_base
    return {"err_baseline": err_base, "err_after": err_perturbed,
            "delta_err": delta_err}


def randomization(model, samples, device,
                  window_width: int = 14) -> Dict:
    means_base, targets = _build_baseline_predictions(model, samples, device, None)
    r2_base = _r2(targets, means_base)
    rng = np.random.default_rng(0)
    perturbed_samples = []
    for s in samples:
        s2 = dict(s)
        X = s["X"].copy()
        mask = _select_window_mask(X.shape[0], s["sport"], window_width)
        idx = np.where(mask)[0]
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        X[idx] = X[shuffled]
        s2["X"] = X
        perturbed_samples.append(s2)
    means_perturbed, _ = _build_baseline_predictions(model, perturbed_samples,
                                                      device, None)
    r2_perturbed = _r2(targets, means_perturbed)
    delta = r2_perturbed - r2_base
    return {"r2_baseline": r2_base, "r2_after": r2_perturbed,
            "delta_R2": delta}


def _r2(y, yhat):
    ss_res = float(np.sum(np.square(y - yhat)))
    ss_tot = float(np.sum(np.square(y - np.mean(y))))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot
