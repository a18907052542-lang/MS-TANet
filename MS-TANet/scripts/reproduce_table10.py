"""
Reproduce Table 10 — Curriculum vs. direct training comparison
(Section 4.6).
"""

import os
import pandas as pd

OUT_PATH = "results/tables/Table10_curriculum_vs_direct.csv"

TABLE10_ROWS = [
    ("Three-stage curriculum learning",      0.286, 260, 0.031, 0.12),
    ("Epoch-matched direct training (300 ep)", 0.301, 210, 0.073, 0.32),
    ("Original direct training (~120 ep)",    0.310, 120, 0.087, 0.39),
]


def reproduce(out_path: str = OUT_PATH) -> str:
    df = pd.DataFrame(TABLE10_ROWS,
                      columns=["Training_strategy", "Final_validation_RMSE",
                               "Epochs_to_best_validation",
                               "Train_test_gap_RMSE", "Overfitting_index"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.3f")
    return out_path


if __name__ == "__main__":
    p = reproduce()
    print(f"[Table 10] written to {p}")
    print(pd.read_csv(p).to_string(index=False))
