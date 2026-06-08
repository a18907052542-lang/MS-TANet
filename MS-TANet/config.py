"""
config.py — Hyperparameter configuration for MS-TANet
Values mirror Table 2 of the paper "Dynamic Correlation Modeling and Algorithm
Optimization of Athletes' Training Load and Performance Based on Deep Time
Series Network".
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class ModelConfig:
    # Sequence
    look_back_window: int = 90              # L
    feature_dim: int = 18                   # D
    static_feature_dim_static: int = 6      # Tier 1: sport, position, sex, age, dominant side, years
    static_feature_dim_dyn: int = 4         # Tier 2: body mass, body-fat %, VO2max, resting HR
    embedding_dim: int = 32                 # D_e
    hidden_dim: int = 128                   # D_h
    # Multi-Scale Dilated Causal Convolution
    conv_kernel_size: int = 3               # K
    dilation_rates: List[int] = field(default_factory=lambda: [1, 7, 28])
    # Sparse Decay Self-Attention
    num_attention_heads: int = 4
    top_k_sparsity: int = 20                # k
    lsh_hyperplanes: int = 4                # H
    initial_decay_lambda: float = 0.04      # initial value of λ
    # Regularization
    dropout: float = 0.2
    physics_alpha_max: float = 0.5
    huber_delta: float = 1.0
    delta_max_default: float = 6.0          # one-day standardised performance bound


@dataclass
class TrainingConfig:
    batch_size: int = 32
    total_epochs: int = 300
    stage1_epochs: int = 100
    stage2_epochs: int = 80
    stage3_epochs: int = 120
    learning_rate_stage1: float = 1e-3
    learning_rate_stage2: float = 5e-4
    learning_rate_stage3: float = 1e-4
    weight_decay: float = 1e-5
    early_stopping_patience: int = 30
    random_seed: int = 42
    n_seeds: int = 50                       # paper repeats with 50 seeds
    n_folds: int = 5                        # 5-fold blocked TS-CV
    deep_ensemble_M: int = 5                # M ensemble members
    train_pct: float = 0.70
    val_pct: float = 0.15
    test_pct: float = 0.15


@dataclass
class EvaluationConfig:
    metrics: List[str] = field(default_factory=lambda: ["RMSE", "MAE", "R2", "NRMSE"])
    physiological_ranges: Dict[str, tuple] = field(default_factory=lambda: {
        "Swimming": (15, 25),
        "Distance Running": (18, 30),
        "Rowing": (16, 28),
    })
    ablation_variants: List[str] = field(default_factory=lambda: [
        "full",
        "no_multi_scale",
        "no_sparse_decay_attention",
        "no_individual_embedding",
        "no_physics_constraint",
        "no_curriculum_learning",
        "no_decay_and_no_individual",
        "param_matched_widened_single_scale",
    ])


MODEL = ModelConfig()
TRAINING = TrainingConfig()
EVAL = EvaluationConfig()


DATA_PATHS = {
    "athlete_cohort": "data/athlete_cohort.csv",
    "daily_training_records": "data/daily_training_records.csv",
    "monthly_test_results": "data/monthly_test_results.csv",
}

SPORTS = ["Swimming", "Distance Running", "Rowing"]

# IEEE color palette used across all figures
IEEE_COLORS = {
    "blue":     "#0072BD",
    "red":      "#D95319",
    "yellow":   "#EDB120",
    "purple":   "#7E2F8E",
    "green":    "#77AC30",
    "cyan":     "#4DBEEE",
    "darkred":  "#A2142F",
    "grey":     "#7F7F7F",
    "navy":     "#1F4E79",
    "orange":   "#ED7D31",
}
