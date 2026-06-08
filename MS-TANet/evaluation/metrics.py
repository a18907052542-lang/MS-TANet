"""
Evaluation metrics — RMSE, MAE, R2, NRMSE (Section 3.7.6).
"""

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum(np.square(y_true - y_pred)))
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rng = float(y_true.max() - y_true.min())
    if rng < 1e-12:
        return 0.0
    return 100.0 * rmse(y_true, y_pred) / rng


def empirical_coverage(y_true: np.ndarray,
                       lower: np.ndarray,
                       upper: np.ndarray) -> float:
    """Proportion of true values that fall inside [lower, upper]."""
    in_band = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(in_band))


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "NRMSE": nrmse(y_true, y_pred),
    }


def cv_summary(values_per_group: dict) -> float:
    """Coefficient of variation (%) of a metric across groups (Section 4.2)."""
    arr = np.array(list(values_per_group.values()), dtype=float)
    if arr.mean() < 1e-12:
        return 0.0
    return 100.0 * arr.std() / arr.mean()
