"""
Three-stage curriculum learning strategy (Section 3.6, Table 1).

Stage 1  (epochs 1-100):  D = 1 (sRPE only), LR = 1e-3, α = 0,
                          IE-FM OFF, all layers trainable.
Stage 2  (epochs 101-180): D = D_full, LR = 5e-4, α = 0,
                          freeze the low-level conv weights, fine-tune higher layers.
Stage 3  (epochs 181-300): D = D_full, LR = 1e-4, α cosine warm-up 0→0.5,
                          IE-FM ON, all parameters thawed, end-to-end joint optimisation.
"""

import math
import numpy as np


class CurriculumSchedule:

    def __init__(self,
                 stage1_epochs: int = 100,
                 stage2_epochs: int = 80,
                 stage3_epochs: int = 120,
                 lr_stage1: float = 1e-3,
                 lr_stage2: float = 5e-4,
                 lr_stage3: float = 1e-4,
                 alpha_max: float = 0.5):
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.stage3_epochs = stage3_epochs
        self.lr_stage1 = lr_stage1
        self.lr_stage2 = lr_stage2
        self.lr_stage3 = lr_stage3
        self.alpha_max = alpha_max
        self.total = stage1_epochs + stage2_epochs + stage3_epochs

    def stage(self, epoch: int) -> int:
        if epoch < self.stage1_epochs:
            return 1
        if epoch < self.stage1_epochs + self.stage2_epochs:
            return 2
        return 3

    def learning_rate(self, epoch: int) -> float:
        s = self.stage(epoch)
        return {1: self.lr_stage1, 2: self.lr_stage2, 3: self.lr_stage3}[s]

    def alpha(self, epoch: int) -> float:
        """Cosine warm-up of α from 0 to α_max throughout Stage 3."""
        if epoch < self.stage1_epochs + self.stage2_epochs:
            return 0.0
        ep_in_s3 = epoch - (self.stage1_epochs + self.stage2_epochs)
        progress = ep_in_s3 / max(1, self.stage3_epochs - 1)
        return 0.5 * self.alpha_max * (1.0 - math.cos(math.pi * progress))

    def use_full_features(self, epoch: int) -> bool:
        return self.stage(epoch) >= 2

    def use_ie_fm(self, epoch: int) -> bool:
        return self.stage(epoch) >= 3

    def physics_active(self, epoch: int) -> bool:
        return self.stage(epoch) >= 3

    def freeze_low_level(self, epoch: int) -> bool:
        return self.stage(epoch) == 2

    def describe(self) -> str:
        return (
            f"Curriculum: Stage 1 ({self.stage1_epochs} ep, LR={self.lr_stage1}) → "
            f"Stage 2 ({self.stage2_epochs} ep, LR={self.lr_stage2}) → "
            f"Stage 3 ({self.stage3_epochs} ep, LR={self.lr_stage3}, "
            f"α 0→{self.alpha_max}). Total {self.total} epochs."
        )


def cosine_lr_schedule(epoch: int, total_epochs: int,
                       lr_max: float = 1e-3, lr_min: float = 1e-5) -> float:
    """Cosine LR schedule used by the epoch-matched direct-training control."""
    progress = epoch / max(1, total_epochs - 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))
