"""
Trainer for MS-TANet with curriculum learning and physics-constrained loss.
"""

import os
import time
import copy
import math
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.curriculum import CurriculumSchedule, cosine_lr_schedule
from training.dataset import AthleteDataset, DAILY_FEATURES
from ms_tanet.physics_loss import PhysicsConstrainedLoss


class Trainer:

    def __init__(self,
                 model: torch.nn.Module,
                 train_samples,
                 val_samples,
                 test_samples=None,
                 batch_size: int = 32,
                 device: str = "cpu",
                 loss_module: Optional[torch.nn.Module] = None,
                 curriculum: Optional[CurriculumSchedule] = None,
                 weight_decay: float = 1e-5,
                 early_stop_patience: int = 30,
                 max_epochs: int = 300,
                 use_curriculum: bool = True,
                 epoch_matched_direct: bool = False,
                 verbose: bool = False):
        self.model = model.to(device)
        self.device = device
        self.train_loader = DataLoader(AthleteDataset(train_samples),
                                       batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(AthleteDataset(val_samples),
                                     batch_size=batch_size, shuffle=False)
        self.test_loader = (DataLoader(AthleteDataset(test_samples),
                                       batch_size=batch_size, shuffle=False)
                            if test_samples else None)
        self.loss_module = loss_module or PhysicsConstrainedLoss(alpha=0.5)
        self.curriculum = curriculum or CurriculumSchedule()
        self.weight_decay = weight_decay
        self.early_stop_patience = early_stop_patience
        self.max_epochs = max_epochs
        self.use_curriculum = use_curriculum
        self.epoch_matched_direct = epoch_matched_direct
        self.verbose = verbose

        # Defer optimiser instantiation — LR changes by stage
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.curriculum.lr_stage1,
            weight_decay=weight_decay,
        )
        self.history = {"train_loss": [], "val_loss": [],
                        "val_rmse": [], "train_rmse": [],
                        "alpha": [], "lr": [], "stage": []}

    # ------------------------------------------------------------------
    def _apply_curriculum(self, epoch: int):
        if self.epoch_matched_direct:
            # Cosine LR decay 1e-3 → 1e-5 over full budget
            lr = cosine_lr_schedule(epoch, self.max_epochs, 1e-3, 1e-5)
            for g in self.optimizer.param_groups:
                g["lr"] = lr
            return lr, 0.0, 3  # always in stage 3 mode (full features)
        if not self.use_curriculum:
            for g in self.optimizer.param_groups:
                g["lr"] = self.curriculum.lr_stage1
            return self.curriculum.lr_stage1, 0.0, 3

        stage = self.curriculum.stage(epoch)
        lr = self.curriculum.learning_rate(epoch)
        alpha = self.curriculum.alpha(epoch)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.loss_module.alpha = alpha

        # Freeze / unfreeze the conv low-level layers in stage 2
        if hasattr(self.model, "ms_dcc"):
            if self.curriculum.freeze_low_level(epoch):
                self.model.ms_dcc.freeze_low_level()
            else:
                self.model.ms_dcc.unfreeze_all()
        return lr, alpha, stage

    # ------------------------------------------------------------------
    def _select_input(self, X: torch.Tensor, stage: int) -> torch.Tensor:
        """Stage 1 uses a single load feature (sRPE) only."""
        if stage == 1 and not self.epoch_matched_direct:
            # Replace all but the sRPE column with zeros
            srpe_idx = DAILY_FEATURES.index("Session_RPE_0_10")
            mask = torch.zeros(X.shape[-1], device=X.device)
            mask[srpe_idx] = 1.0
            return X * mask
        return X

    # ------------------------------------------------------------------
    def train(self) -> dict:
        best_val_rmse = float("inf")
        best_state = None
        patience = 0
        t0 = time.time()

        for epoch in range(self.max_epochs):
            lr, alpha, stage = self._apply_curriculum(epoch)
            self.model.train()
            train_losses, train_residuals = [], []

            for X, s_static, s_dyn, y, _ in self.train_loader:
                X = X.to(self.device); s_static = s_static.to(self.device)
                s_dyn = s_dyn.to(self.device); y = y.to(self.device)
                X = self._select_input(X, stage)

                mean, logvar = self.model(X, s_static, s_dyn)
                attn = getattr(self.model, "_last_attn_weights", None)
                lam = getattr(self.model, "decay_lambda", None)
                phys_active = self.curriculum.physics_active(epoch) and not self.epoch_matched_direct
                loss_dict = self.loss_module(
                    mean, logvar, y,
                    decay_lambda=lam, attention_weights=attn,
                    phys_active=phys_active,
                )
                self.optimizer.zero_grad()
                loss_dict["L_total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                train_losses.append(loss_dict["L_total"].item())
                train_residuals.extend((y - mean).detach().cpu().numpy().tolist())

            train_rmse = float(np.sqrt(np.mean(np.square(train_residuals))))

            # Validation
            val_loss, val_rmse = self.evaluate(self.val_loader, stage)
            self.history["train_loss"].append(float(np.mean(train_losses)))
            self.history["val_loss"].append(val_loss)
            self.history["train_rmse"].append(train_rmse)
            self.history["val_rmse"].append(val_rmse)
            self.history["alpha"].append(alpha)
            self.history["lr"].append(lr)
            self.history["stage"].append(stage)

            if self.verbose and epoch % 10 == 0:
                print(f"[ep {epoch:3d} stage {stage}] LR={lr:.4g} α={alpha:.3f} "
                      f"train_rmse={train_rmse:.4f} val_rmse={val_rmse:.4f}")

            if val_rmse < best_val_rmse - 1e-4:
                best_val_rmse = val_rmse
                best_state = copy.deepcopy(self.model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stop_patience and stage == 3:
                    if self.verbose:
                        print(f"  early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {
            "best_val_rmse": best_val_rmse,
            "history": self.history,
            "wall_time_s": time.time() - t0,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, stage: int = 3):
        self.model.eval()
        losses, residuals = [], []
        for X, s_static, s_dyn, y, _ in loader:
            X = X.to(self.device); s_static = s_static.to(self.device)
            s_dyn = s_dyn.to(self.device); y = y.to(self.device)
            X = self._select_input(X, stage)
            mean, logvar = self.model(X, s_static, s_dyn)
            attn = getattr(self.model, "_last_attn_weights", None)
            lam = getattr(self.model, "decay_lambda", None)
            loss_dict = self.loss_module(
                mean, logvar, y,
                decay_lambda=lam, attention_weights=attn,
                phys_active=False,
            )
            losses.append(loss_dict["L_total"].item())
            residuals.extend((y - mean).cpu().numpy().tolist())
        rmse = float(np.sqrt(np.mean(np.square(residuals))))
        return float(np.mean(losses)), rmse

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, loader: DataLoader):
        self.model.eval()
        means, logvars, targets, ids = [], [], [], []
        for X, s_static, s_dyn, y, idx in loader:
            X = X.to(self.device); s_static = s_static.to(self.device)
            s_dyn = s_dyn.to(self.device)
            mean, logvar = self.model(X, s_static, s_dyn)
            means.extend(mean.cpu().numpy().tolist())
            logvars.extend(logvar.cpu().numpy().tolist())
            targets.extend(y.numpy().tolist())
            ids.extend(idx.numpy().tolist())
        return (np.array(means), np.array(logvars),
                np.array(targets), np.array(ids))
