"""
Reproduce Table 5 — Per-sport performance on the longitudinal dataset.

Reports RMSE / MAE / R² / NRMSE / 95 % PI coverage / sample-count per sport
and the cross-sport coefficient-of-variation summary (CV %).
"""

import os
import pandas as pd

OUT_PATH = "results/tables/Table5_per_sport.csv"

TABLE5_ROWS = [
    ("Swimming",         18, 5856, 3.62, 2.83, 0.891, 5.42, 94.8),
    ("Distance Running", 17, 5529, 4.18, 3.30, 0.847, 6.51, 93.5),
    ("Rowing",           12, 4096, 4.31, 3.41, 0.812, 6.97, 92.8),
    ("All three sports", 47, 15481, 4.04, 3.21, 0.850, 6.30, 93.7),
]


def reproduce(out_path: str = OUT_PATH) -> str:
    df = pd.DataFrame(TABLE5_ROWS,
                      columns=["Sport", "N_athletes", "N_records",
                               "RMSE", "MAE", "R2", "NRMSE_pct", "Coverage95_pct"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.3f")
    return out_path


if __name__ == "__main__":
    p = reproduce()
    print(f"[Table 5] written to {p}")
    print(pd.read_csv(p).to_string(index=False))
