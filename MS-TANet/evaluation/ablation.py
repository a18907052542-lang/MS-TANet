"""
Ablation study — Section 4.3, Table 7.

Seven ablation variants:
  1. Full MS-TANet
  2. w/o Multi-Scale Conv (single-scale d=1)
  3. w/o Sparse Decay Attention (standard attention)
  4. w/o Individual Embedding Module
  5. w/o Physics-Constrained Regularization (α = 0)
  6. w/o Curriculum Learning (direct end-to-end)
  7. w/o Decay Attention + Individual Embedding
  8. Parameter-matched widened single-scale (178 ch) — capacity control
"""

from typing import Callable, Dict, List
import copy
import numpy as np
import torch

from ms_tanet.model import MSTANet
from training.curriculum import CurriculumSchedule
from evaluation.cross_validation import run_chronological_cv


def build_variant(name: str, base_kwargs: dict):
    """Return a model_builder callable for the requested ablation variant."""
    kwargs = dict(base_kwargs)
    use_curriculum = True
    alpha_max = 0.5

    if name == "full":
        pass
    elif name == "no_multi_scale":
        kwargs["use_multi_scale"] = False
    elif name == "no_sparse_decay_attention":
        kwargs["use_sparse_decay_attention"] = False
    elif name == "no_individual_embedding":
        kwargs["use_individual_embedding"] = False
    elif name == "no_physics_constraint":
        alpha_max = 0.0
    elif name == "no_curriculum_learning":
        use_curriculum = False
    elif name == "no_decay_and_no_individual":
        kwargs["use_sparse_decay_attention"] = False
        kwargs["use_individual_embedding"] = False
    elif name == "param_matched_widened_single_scale":
        kwargs["use_multi_scale"] = False
        kwargs["widened_single_scale_channels"] = 178
    else:
        raise ValueError(f"Unknown ablation variant {name}")

    def builder():
        return MSTANet(**kwargs)
    return builder, use_curriculum, alpha_max


def run_full_ablation(samples,
                      model_kwargs: dict,
                      device: str = "cpu",
                      max_epochs: int = 30) -> Dict:
    """Run the seven ablation variants under 5-fold blocked TS-CV."""
    variants = [
        "full",
        "no_multi_scale",
        "no_sparse_decay_attention",
        "no_individual_embedding",
        "no_physics_constraint",
        "no_curriculum_learning",
        "no_decay_and_no_individual",
        "param_matched_widened_single_scale",
    ]
    results = {}
    for v in variants:
        builder, use_curr, alpha_max = build_variant(v, model_kwargs)
        curriculum = CurriculumSchedule(alpha_max=alpha_max) if use_curr else None
        res = run_chronological_cv(
            builder, samples,
            device=device, max_epochs=max_epochs,
            curriculum=curriculum, use_curriculum=use_curr,
            alpha_max=alpha_max,
        )
        results[v] = res
        # Trainable parameter count
        results[v]["num_params_M"] = builder().num_parameters() / 1e6
    return results


def count_parameters(model_kwargs: dict, variant: str = "full") -> float:
    builder, _, _ = build_variant(variant, model_kwargs)
    model = builder()
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
