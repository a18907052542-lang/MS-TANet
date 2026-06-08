"""
Main training entry-point.

Trains MS-TANet end-to-end on the longitudinal dataset with the three-stage
curriculum and physics-constrained loss, then evaluates on the held-out test
split.  All hyperparameters are loaded from config.py.

Usage:
    python -m scripts.train_main_experiment           # default
    python -m scripts.train_main_experiment --epochs 30 --device cpu --seed 7
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL, TRAINING, DATA_PATHS
from ms_tanet.model import MSTANet
from ms_tanet.physics_loss import PhysicsConstrainedLoss
from training.dataset import AthleteSampleIndex, chronological_split
from training.curriculum import CurriculumSchedule
from training.trainer import Trainer
from torch.utils.data import DataLoader
from training.dataset import AthleteDataset
from evaluation.metrics import all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=TRAINING.total_epochs)
    parser.add_argument("--batch-size", type=int, default=TRAINING.batch_size)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=TRAINING.random_seed)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint", default="checkpoints/ms_tanet_final.pt")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.time()
    print(f"[1/4] Loading data from {args.data_dir}")
    idx = AthleteSampleIndex(data_dir=args.data_dir, look_back=MODEL.look_back_window)
    print(f"      {len(idx.samples)} supervised samples constructed "
          f"({len({s['athlete_id'] for s in idx.samples})} athletes).")

    print("[2/4] Building per-athlete chronological splits")
    train, val, test = chronological_split(idx.samples,
                                            train_pct=TRAINING.train_pct,
                                            val_pct=TRAINING.val_pct,
                                            test_pct=TRAINING.test_pct)
    print(f"      train={len(train)}  val={len(val)}  test={len(test)}")

    print("[3/4] Constructing MS-TANet")
    model = MSTANet(
        feature_dim=MODEL.feature_dim,
        static_dim_static=MODEL.static_feature_dim_static,
        static_dim_dyn=MODEL.static_feature_dim_dyn,
        hidden_dim=MODEL.hidden_dim,
        embedding_dim=MODEL.embedding_dim,
        dilation_rates=MODEL.dilation_rates,
        conv_kernel_size=MODEL.conv_kernel_size,
        num_attention_heads=MODEL.num_attention_heads,
        top_k=MODEL.top_k_sparsity,
        lsh_hyperplanes=MODEL.lsh_hyperplanes,
        initial_lambda=MODEL.initial_decay_lambda,
        dropout=MODEL.dropout,
    )
    n_params = model.num_parameters()
    print(f"      MS-TANet has {n_params/1e6:.2f} M trainable parameters")

    curriculum = CurriculumSchedule(
        stage1_epochs=TRAINING.stage1_epochs,
        stage2_epochs=TRAINING.stage2_epochs,
        stage3_epochs=TRAINING.stage3_epochs,
        lr_stage1=TRAINING.learning_rate_stage1,
        lr_stage2=TRAINING.learning_rate_stage2,
        lr_stage3=TRAINING.learning_rate_stage3,
        alpha_max=MODEL.physics_alpha_max,
    )
    print(f"      {curriculum.describe()}")

    loss = PhysicsConstrainedLoss(
        huber_delta=MODEL.huber_delta,
        delta_max=MODEL.delta_max_default,
        alpha=MODEL.physics_alpha_max,
    )

    print("[4/4] Training with three-stage curriculum")
    trainer = Trainer(
        model, train, val, test,
        batch_size=args.batch_size,
        device=args.device,
        loss_module=loss,
        curriculum=curriculum,
        weight_decay=TRAINING.weight_decay,
        early_stop_patience=TRAINING.early_stopping_patience,
        max_epochs=args.epochs,
        verbose=args.verbose,
    )
    train_result = trainer.train()

    # Evaluate on test split
    test_loader = DataLoader(AthleteDataset(test), batch_size=args.batch_size,
                             shuffle=False)
    means, _, targets, _ = trainer.predict(test_loader)
    metrics = all_metrics(targets, means)

    print("\n" + "=" * 60)
    print("Final test-set performance")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:<8s} = {v:.4f}")
    print(f"\nWall-clock training time : {train_result['wall_time_s']:.1f} s")
    print(f"Best validation RMSE     : {train_result['best_val_rmse']:.4f}")

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "config": {"model": MODEL.__dict__, "training": TRAINING.__dict__,
                   "epochs_run": len(train_result["history"]["val_rmse"])},
    }, args.checkpoint)
    print(f"\nCheckpoint saved to {args.checkpoint}")

    # Save metrics for downstream use
    metrics_path = os.path.join("results", "tables", "main_training_metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({"metrics": metrics,
                   "wall_time_s": train_result["wall_time_s"]}, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    print(f"Total wall time          : {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
