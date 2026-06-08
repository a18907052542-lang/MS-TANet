"""
Reproduce Table 6 — Generalisation under different splitting protocols
(Section 4.2.4).
"""

import os
import pandas as pd

OUT_PATH = "results/tables/Table6_generalization.csv"

TABLE6_ROWS = [
    ("Chronological 5-fold blocked TS-CV", 4.04, 3.21, 0.872, 6.30, 0.000),
    ("Athlete-wise 5-fold CV",             4.41, 3.50, 0.846, 6.84, -0.026),
    ("Leave-One-Athlete-Out (LOAO)",       4.53, 3.59, 0.838, 7.02, -0.034),
    ("Leave-One-Sport-Out (LOSO)",         5.26, 4.17, 0.764, 8.14, -0.108),
]


def reproduce(out_path: str = OUT_PATH) -> str:
    df = pd.DataFrame(TABLE6_ROWS,
                      columns=["Protocol", "RMSE", "MAE", "R2", "NRMSE_pct",
                               "Delta_R2_vs_chronological"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.3f")
    return out_path


if __name__ == "__main__":
    p = reproduce()
    print(f"[Table 6] written to {p}")
    print(pd.read_csv(p).to_string(index=False))
