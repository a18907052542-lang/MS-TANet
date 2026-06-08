"""
Cross-validation runners — Section 3.7.3 and Section 4.2 (Table 6).

Five protocols supported:
  * Chronological 5-fold blocked TS-CV (default for Tables 3 and 4).
  * Athlete-wise CV (random partition of athletes within sport, n=5 folds).
  * Leave-One-Athlete-Out (LOAO) — train on 46, test on 1.
  * Leave-One-Sport-Out (LOSO) — train on 2 sports, test on the third.
  * Deep-ensemble (M = 5) — wraps any protocol with multiple seed initialisations.
"""

from __future__ import annotations
import math
import copy
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Callable, Dict, List, Tuple

from training.dataset import (
    AthleteDataset,
    blocked_ts_cv_folds, athlete_wise_cv,
    leave_one_athlete_out, leave_one_sport_out,
)
from training.curriculum import CurriculumSchedule
from training.trainer import Trainer
from ms_tanet.physics_loss import PhysicsConstrainedLoss
from evaluation.metrics import rmse, mae, r2, nrmse, all_metrics, empirical_coverage


# ----------------------------------------------------------------------
def _run_single_fold(model_builder: Callable[[], torch.nn.Module],
                     train_samples, val_samples, test_samples,
                     device: str = "cpu",
                     curriculum: CurriculumSchedule = None,
                     use_curriculum: bool = True,
                     epoch_matched_direct: bool = False,
                     max_epochs: int = 300,
                     batch_size: int = 32,
                     n_ensemble: int = 1,
                     seeds: List[int] = (0,),
                     alpha_max: float = 0.5):
    """Train one fold; return predictions on test split (with deep-ensemble)."""
    test_loader = DataLoader(AthleteDataset(test_samples),
                             batch_size=batch_size, shuffle=False)
    ensemble_means, ensemble_logvars = [], []
    for seed in seeds[:n_ensemble]:
        torch.manual_seed(seed); np.random.seed(seed)
        model = model_builder()
        loss = PhysicsConstrainedLoss(alpha=alpha_max)
        trainer = Trainer(
            model, train_samples, val_samples, test_samples,
            batch_size=batch_size, device=device,
            loss_module=loss, curriculum=curriculum,
            max_epochs=max_epochs,
            use_curriculum=use_curriculum,
            epoch_matched_direct=epoch_matched_direct,
        )
        trainer.train()
        means, logvars, targets, _ = trainer.predict(test_loader)
        ensemble_means.append(means)
        ensemble_logvars.append(logvars)
    means = np.mean(np.stack(ensemble_means), axis=0)
    aleatoric = np.mean(np.exp(np.stack(ensemble_logvars)), axis=0)
    epistemic = np.var(np.stack(ensemble_means), axis=0)
    total_var = aleatoric + epistemic
    sigma = np.sqrt(total_var)
    targets = np.array([s["y"] for s in test_samples])
    return {
        "means": means, "logvars": np.log(total_var + 1e-8),
        "targets": targets, "sigma": sigma,
    }


def run_chronological_cv(model_builder, samples, **kwargs) -> Dict:
    folds = blocked_ts_cv_folds(samples)
    all_results = []
    for k, (train, val, test) in enumerate(folds):
        if len(test) == 0:
            continue
        res = _run_single_fold(model_builder, train, val, test, **kwargs)
        m = all_metrics(res["targets"], res["means"])
        m["fold"] = k
        m["coverage_95"] = empirical_coverage(
            res["targets"],
            res["means"] - 1.96 * res["sigma"],
            res["means"] + 1.96 * res["sigma"]
        )
        all_results.append(m)
    return aggregate_fold_metrics(all_results)


def run_athlete_wise_cv(model_builder, samples, n_folds: int = 5, **kwargs) -> Dict:
    folds = athlete_wise_cv(samples, n_folds=n_folds)
    out = []
    for k, (train, val, test) in enumerate(folds):
        if len(test) == 0:
            continue
        res = _run_single_fold(model_builder, train, val, test, **kwargs)
        m = all_metrics(res["targets"], res["means"])
        m["fold"] = k
        out.append(m)
    return aggregate_fold_metrics(out)


def run_leave_one_athlete_out(model_builder, samples, **kwargs) -> Dict:
    folds = leave_one_athlete_out(samples)
    out = []
    for k, (train, val, test) in enumerate(folds):
        if len(test) == 0:
            continue
        res = _run_single_fold(model_builder, train, val, test, **kwargs)
        m = all_metrics(res["targets"], res["means"])
        m["fold"] = k
        out.append(m)
    return aggregate_fold_metrics(out)


def run_leave_one_sport_out(model_builder, samples, **kwargs) -> Dict:
    folds = leave_one_sport_out(samples)
    out = []
    for k, (train, val, test) in enumerate(folds):
        if len(test) == 0:
            continue
        res = _run_single_fold(model_builder, train, val, test, **kwargs)
        m = all_metrics(res["targets"], res["means"])
        m["fold"] = k
        out.append(m)
    return aggregate_fold_metrics(out)


def aggregate_fold_metrics(metric_dicts: List[Dict]) -> Dict:
    keys = ["RMSE", "MAE", "R2", "NRMSE"]
    if "coverage_95" in metric_dicts[0]:
        keys.append("coverage_95")
    agg = {}
    for k in keys:
        vals = np.array([m[k] for m in metric_dicts if k in m])
        agg[k] = {"mean": float(vals.mean()), "std": float(vals.std()),
                  "values": vals.tolist()}
    agg["n_folds"] = len(metric_dicts)
    return agg
