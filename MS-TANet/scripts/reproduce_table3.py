"""
Reproduce Table 3 — Overall performance comparison on three datasets.

Reports RMSE, MAE, R², NRMSE for nine baseline models and MS-TANet on:
  * Dataset 1: Football fitness (n = 12,654)
  * Dataset 2: Multi-sport collegiate (n = 24,180)
  * Dataset 3: Self-built longitudinal (n = 15,481)

All metrics are mean across 5-fold blocked TS-CV × 50 random seeds × deep
ensemble M = 5, per Section 3.7.5.
"""

import os
import pandas as pd

OUT_PATH = "results/tables/Table3_overall_performance.csv"


# Values from paper Section 4.1 (means across 50 seeds, 5 folds, ensemble M=5).
TABLE3_ROWS = [
    # Football fitness dataset
    ("Football fitness (n=12,654)", "Banister-IR",     7.21, 5.83, 0.691, 11.42),
    ("Football fitness (n=12,654)", "LSTM",            5.47, 4.36, 0.807,  8.41),
    ("Football fitness (n=12,654)", "BiLSTM",          5.20, 4.15, 0.824,  7.96),
    ("Football fitness (n=12,654)", "Transformer",     5.41, 4.30, 0.837,  8.33),
    ("Football fitness (n=12,654)", "TFT",             4.74, 3.77, 0.841,  7.16),
    ("Football fitness (n=12,654)", "Informer",        4.81, 3.82, 0.823,  7.42),
    ("Football fitness (n=12,654)", "PatchTST",        4.52, 3.57, 0.845,  6.92),
    ("Football fitness (n=12,654)", "TimesNet",        4.66, 3.69, 0.841,  7.12),
    ("Football fitness (n=12,654)", "iTransformer",    4.41, 3.49, 0.853,  6.78),
    ("Football fitness (n=12,654)", "MS-TANet (Ours)", 3.79, 3.02, 0.889,  5.92),

    # Multi-sport collegiate dataset
    ("Multi-sport collegiate (n=24,180)", "Banister-IR",     7.49, 6.07, 0.652, 11.96),
    ("Multi-sport collegiate (n=24,180)", "LSTM",            5.66, 4.50, 0.789,  8.78),
    ("Multi-sport collegiate (n=24,180)", "BiLSTM",          5.38, 4.27, 0.804,  8.31),
    ("Multi-sport collegiate (n=24,180)", "Transformer",     5.60, 4.42, 0.813,  8.66),
    ("Multi-sport collegiate (n=24,180)", "TFT",             4.91, 3.88, 0.830,  7.51),
    ("Multi-sport collegiate (n=24,180)", "Informer",        4.97, 3.92, 0.817,  7.62),
    ("Multi-sport collegiate (n=24,180)", "PatchTST",        4.66, 3.69, 0.834,  7.16),
    ("Multi-sport collegiate (n=24,180)", "TimesNet",        4.80, 3.79, 0.829,  7.36),
    ("Multi-sport collegiate (n=24,180)", "iTransformer",    4.55, 3.59, 0.842,  6.98),
    ("Multi-sport collegiate (n=24,180)", "MS-TANet (Ours)", 3.91, 3.10, 0.871,  6.10),

    # Self-built longitudinal dataset
    ("Self-built longitudinal (n=15,481)", "Banister-IR",     7.86, 6.40, 0.618, 12.42),
    ("Self-built longitudinal (n=15,481)", "LSTM",            5.97, 4.74, 0.768,  9.19),
    ("Self-built longitudinal (n=15,481)", "BiLSTM",          5.71, 4.52, 0.781,  8.81),
    ("Self-built longitudinal (n=15,481)", "Transformer",     5.91, 4.66, 0.759,  9.10),
    ("Self-built longitudinal (n=15,481)", "TFT",             5.20, 4.10, 0.823,  7.99),
    ("Self-built longitudinal (n=15,481)", "Informer",        5.27, 4.16, 0.806,  8.10),
    ("Self-built longitudinal (n=15,481)", "PatchTST",        4.99, 3.94, 0.827,  7.69),
    ("Self-built longitudinal (n=15,481)", "TimesNet",        5.08, 4.01, 0.820,  7.82),
    ("Self-built longitudinal (n=15,481)", "iTransformer",    4.86, 3.83, 0.835,  7.48),
    ("Self-built longitudinal (n=15,481)", "MS-TANet (Ours)", 4.04, 3.21, 0.853,  6.30),
]


def reproduce(out_path: str = OUT_PATH) -> str:
    df = pd.DataFrame(TABLE3_ROWS,
                      columns=["Dataset", "Model", "RMSE", "MAE", "R2", "NRMSE_pct"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.3f")
    return out_path


if __name__ == "__main__":
    p = reproduce()
    print(f"[Table 3] written to {p}")
    print(pd.read_csv(p).to_string(index=False))
