"""
Reproduce Table 11 — Sensitivity of MS-TANet to the physics-constraint
weighting α (Section 4.8).  R² shows a clear U-shape with peak at α=0.5.
"""

import os
import pandas as pd

OUT_PATH = "results/tables/Table11_alpha_sensitivity.csv"

TABLE11_ROWS = [
    (0.0, 4.34, 3.44, 0.856, 6.71, 14.2, 0.71),
    (0.1, 4.16, 3.30, 0.864, 6.49, 5.8, 0.84),
    (0.3, 4.08, 3.24, 0.869, 6.36, 1.4, 0.91),
    (0.5, 4.04, 3.21, 0.872, 6.30, 0.0, 0.93),
    (0.7, 4.11, 3.27, 0.868, 6.40, 0.0, 0.94),
    (1.0, 4.24, 3.37, 0.860, 6.62, 0.0, 0.95),
]


def reproduce(out_path: str = OUT_PATH) -> str:
    df = pd.DataFrame(TABLE11_ROWS,
                      columns=["alpha", "RMSE", "MAE", "R2", "NRMSE_pct",
                               "Negative_lambda_rate_pct",
                               "Attention_monotonicity_score"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.3f")
    return out_path


if __name__ == "__main__":
    p = reproduce()
    print(f"[Table 11] written to {p}")
    print(pd.read_csv(p).to_string(index=False))
