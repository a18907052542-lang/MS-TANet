# MS-TANet — Multi-Scale Temporal Attention Network

Reference implementation accompanying the paper
**"Dynamic Correlation Modeling and Algorithm Optimization of Athletes'
Training Load and Performance Based on Deep Time Series Network."**

MS-TANet captures heterogeneous-time-scale local periodicity, long-range
delayed adaptation, and individual physiological differences in athletic
training-load → performance forecasting.  The architecture is composed of
three complementary modules — multi-scale dilated causal convolution
(**MS-DCC**), sparse decay self-attention (**SD-SA**), and individual
embedding & feature modulation (**IE-FM**) — trained end-to-end under a
three-stage curriculum with a physics-constrained regulariser.

---

## 1. Project Layout

```
MS-TANet/
├── config.py                       # All hyper-parameters (paper Table 2)
├── requirements.txt
├── README.md
│
├── data/                           # Experimental data tables
│   ├── athlete_cohort.csv             — 47 athletes × static / quasi-dynamic attributes
│   ├── daily_training_records.csv     — 15,481 athlete-day records (18 features)
│   └── monthly_test_results.csv       — 846 monthly performance tests
│
├── ms_tanet/                       # Core network modules
│   ├── ms_dcc.py                      — Multi-Scale Dilated Causal Conv (Eq. 2–3)
│   ├── sd_sa.py                       — Sparse Decay Self-Attention + LSH (Eq. 4–5)
│   ├── ie_fm.py                       — Individual Embedding + Feature Modulation (Eq. 6–7)
│   ├── model.py                       — Full MSTANet model
│   ├── baselines.py                   — Banister-IR, LSTM, BiLSTM, Transformer, TFT,
│   │                                    Informer, PatchTST, TimesNet, iTransformer
│   └── physics_loss.py                — Physics-constrained loss (Eq. 8–13)
│
├── training/                       # Training infrastructure
│   ├── dataset.py                     — AthleteSampleIndex, splits, 5-fold blocked TS-CV
│   ├── curriculum.py                  — 3-stage curriculum (Table 1)
│   └── trainer.py                     — Trainer with curriculum + early stopping
│
├── evaluation/                     # Evaluation primitives
│   ├── metrics.py                     — RMSE / MAE / R² / NRMSE / 95 % PI coverage
│   ├── cross_validation.py            — TS-CV / athlete-wise / LOAO / LOSO
│   ├── ablation.py                    — Seven-variant ablation runner
│   ├── shapley.py                     — Shapley decomposition over 2⁵ coalitions
│   ├── sensitivity.py                 — α sensitivity sweep
│   └── attention_analysis.py          — Window removal / perturbation / randomisation
│
├── visualization/                  # Publication-quality figures
│   ├── style.py                       — IEEE matplotlib styling
│   ├── figure_8_radar.py              — Multi-dim radar (10 models)
│   ├── figure_9_scatter.py            — Observed vs. predicted with 95 % PI
│   ├── figure_10_shapley.py           — Shapley donut + ranked bar
│   ├── figure_11_heatmap.py           — Attention heatmap by sport
│   ├── figure_12_decay.py             — Per-sport exponential decay curves
│   ├── figure_13_gating.py            — Daily + population gating dynamics
│   └── figure_14_convergence.py       — Curriculum vs direct-training curves
│
├── scripts/                        # Reproduction entry-points
│   ├── train_main_experiment.py       — End-to-end training driver
│   ├── reproduce_table3.py … table12.py   — One script per result table
│   └── reproduce_all.py               — Master script: all tables + all figures
│
├── results/
│   ├── tables/                        — Result CSVs
│   └── figures/                       — Publication PNGs (≥ 600 DPI)
└── checkpoints/                       — Trained model checkpoints
```

---

## 2. Quick Start

```bash
# (a) Install dependencies
pip install -r requirements.txt

# (b) Regenerate every result table and every figure
python scripts/reproduce_all.py

# (c) End-to-end training run (default = 300 epochs, GPU if available)
python scripts/train_main_experiment.py --verbose

# (d) Generate one specific figure
python visualization/figure_8_radar.py
```

After step (b) you will find:

* `results/tables/Table3_overall_performance.csv` — overall RMSE/MAE/R²/NRMSE
  across the three datasets (Section 4.1).
* `results/tables/Table4_sota_comparison.csv` — SOTA comparison vs Informer,
  PatchTST, TimesNet, iTransformer.
* `results/tables/Table5_per_sport.csv` — per-sport breakdown
  (Swimming / Distance Running / Rowing).
* `results/tables/Table6_generalization.csv` — chronological vs athlete-wise
  vs LOAO vs LOSO.
* `results/tables/Table7_ablation.csv` — seven ablation variants.
* `results/tables/Table8_decay_parameters.csv` — learned λ and t₁/₂.
* `results/tables/Table9_population_gating.csv` — phase × scale gating means.
* `results/tables/Table10_curriculum_vs_direct.csv` — convergence comparison.
* `results/tables/Table11_alpha_sensitivity.csv` — α sweep.
* `results/tables/Table12_physics_effect.csv` — interpretability diagnostics.

and seven publication figures in `results/figures/`.

---

## 3. Data Format

### Athlete cohort table (`athlete_cohort.csv`, n = 47 rows)

| Column                                | Type     | Description                              |
| ------------------------------------- | -------- | ---------------------------------------- |
| `Athlete_ID`                          | string   | Anonymised identifier (e.g. `SW-001`)    |
| `Sport`                               | string   | `Swimming`, `Distance Running`, `Rowing` |
| `Sex`                                 | string   | `Male` / `Female`                        |
| `Age_at_season_start_years`           | float    | Age at the season start                  |
| `Height_cm`                           | float    | Standing height                          |
| `Dominant_side`                       | string   | `R` / `L`                                |
| `Years_of_structured_training`        | int      | Cumulative structured-training years     |

### Daily training records (`daily_training_records.csv`, n = 15 481)

18 daily features per athlete-day; columns include:
`Training_duration_min`, `Total_distance_km`, `Mean_HR_bpm`, `Peak_HR_bpm`,
`TRIMP`, `HRV_RMSSD_morning_ms`, `Stroke_count_total`,
`Stroke_rate_per_min`, `PlayerLoad_au`, `Session_RPE_0_10`,
`sRPE_load_AU`, `Prior_night_sleep_h`, `Subjective_wellness_1_10`, and a
one-hot encoding of `Training_phase` ∈ {`Base`, `Intensive`, `Tapering`,
`Competition`, `Recovery`}.

### Monthly test results (`monthly_test_results.csv`, n = 846)

Body composition (mass, fat %), VO₂max, resting HR, plus the standardised
0–100 sport-specific performance score.

---

## 4. Model Architecture

**Eq. (2)** — Causal dilated convolution at scale *m* with dilation *dₘ*:

```
H_m(t) = σ( Σ_{k=0}^{K-1} W_m^{(l,k)} · H(t - dₘ · k) + b_m^{(l)} )
```

**Eq. (3)** — Learnable gating fusion across the three branches:

```
H_fused(t) = Σ_m g_m(t) ⊙ H_m(t),     g(t) = Softmax(W_g [H_1; H_2; H_3])
```

**Eq. (4)** — Decay-biased attention logit:

```
a_{t,t'} = Softmax( (Q_t K_{t'}ᵀ / √D_k) − λ·(t − t') )
```

**Eq. (5)** — Sparse top-k aggregation:

```
SD-SA(Q,K,V) = Σ_{t' ∈ N_k(t)} a_{t,t'} · V_{t'}
```

**Eq. (6)** — Individual embedding from two-tier conditioning:

```
e_i = ReLU(W_s · s_i),   γ_i = σ(W_γ · e_i),   β_i = W_β · e_i
```

**Eq. (7)** — Feature modulation:

```
H_i(t) = γ_i ⊙ H_fused(t) + β_i
```

**Eq. (8–12)** — Physics-constrained training loss:

```
L_total = L_pred + α · L_phys
L_pred  = (1 / NT) Σ ρ_δ(y − ŷ)              (Huber)
L_phys  = L_decay + L_bound
L_decay = ReLU(−λ) + (1/|S|) Σ ReLU(a_{t,t''} − a_{t,t'})
L_bound = (1 / NT) Σ ReLU(|ŷ(t) − ŷ(t−1)| − Δ_max)
```

---

## 5. Three-Stage Curriculum Learning (Table 1)

| Stage | Epochs | LR    | α (physics) | IE-FM | Input dim | Notes                              |
| ----- | ------ | ----- | ----------- | ----- | --------- | ---------------------------------- |
| 1     | 100    | 1e-3  | 0           | OFF   | 1 (sRPE)  | Single-load warm-up                |
| 2     | 80     | 5e-4  | 0           | OFF   | Full D    | Freeze low-level conv; broaden     |
| 3     | 120    | 1e-4  | 0 → 0.5     | ON    | Full D    | Joint optimisation + cosine α-ramp |

Total budget = **300 epochs**.  Early stopping with patience 30 inside stage 3.

---

## 6. Reproducing the Paper

To regenerate **every** quantitative claim with paper-aligned numerics:

```bash
python scripts/reproduce_all.py
```

The script runs all ten `reproduce_tableX.py` modules and renders the seven
data-driven figures.  Each table CSV is written in long-format,
machine-readable form and is directly importable into LaTeX (e.g. via
`\input{Table3.tex}` after a one-line `csvsimple` conversion).

To verify the training pipeline on the included longitudinal data, run:

```bash
python scripts/train_main_experiment.py --epochs 30 --verbose
```

(30-epoch debug run; the full 300-epoch curriculum reproduces the paper's
final R² = 0.853 on this dataset.)

---

## 7. Hardware

* GPU (CUDA): recommended.  Full training takes ≈ 2.5 h on an NVIDIA A100.
* CPU: the 300-epoch curriculum takes ≈ 6 h on a modern server CPU.
* Memory: ≈ 4 GB GPU memory; 8 GB host RAM.

---

## 8. Citation

If you use this code, please cite the paper:

```
@article{ms_tanet_2025,
  title  = {Dynamic Correlation Modeling and Algorithm Optimization of
            Athletes' Training Load and Performance Based on Deep Time
            Series Network},
  author = {Author, A. and Co-author, B.},
  year   = {2025},
  journal = {Journal of XXX}
}
```

---

## 9. License

Code: Apache 2.0.  Data: CC BY-NC 4.0 (research-only use).
