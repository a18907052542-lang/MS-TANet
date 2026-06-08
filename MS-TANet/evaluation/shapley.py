"""
Shapley-value decomposition — Section 4.3 of the paper.

Five core innovations:
  1. Multi-scale dilated causal convolution
  2. Sparse decay self-attention
  3. Individual embedding
  4. Physics constraint
  5. Curriculum learning

The decomposition enumerates the 2^5 = 32 coalitions and computes the
characteristic-function value v(S) = R²(coalition S).  The Shapley value of
component i is
    φ_i = (1/n!) Σ_S (|S|! · (n-|S|-1)!) · [v(S ∪ {i}) - v(S)]

and the pairwise interaction term is
    I(i,j) = (1/2) [v({i,j}) - v({i}) - v({j}) + v(∅)]   (approximation)
"""

from itertools import combinations, chain
from math import factorial
from typing import Dict, List, Callable
import numpy as np

from ms_tanet.model import MSTANet
from training.curriculum import CurriculumSchedule
from evaluation.cross_validation import run_chronological_cv


COMPONENTS = [
    "multi_scale_conv",      # 1
    "sparse_decay_attn",     # 2
    "individual_embedding",  # 3
    "physics_constraint",    # 4
    "curriculum_learning",   # 5
]


def _coalition_to_kwargs(coalition: List[str], base_kwargs: dict):
    kwargs = dict(base_kwargs)
    use_curriculum = True
    alpha_max = 0.5
    if "multi_scale_conv" not in coalition:
        kwargs["use_multi_scale"] = False
    if "sparse_decay_attn" not in coalition:
        kwargs["use_sparse_decay_attention"] = False
    if "individual_embedding" not in coalition:
        kwargs["use_individual_embedding"] = False
    if "physics_constraint" not in coalition:
        alpha_max = 0.0
    if "curriculum_learning" not in coalition:
        use_curriculum = False
    return kwargs, use_curriculum, alpha_max


def evaluate_coalition(coalition: List[str],
                       samples,
                       base_kwargs: dict,
                       device: str = "cpu",
                       max_epochs: int = 30) -> float:
    kwargs, use_curr, alpha_max = _coalition_to_kwargs(coalition, base_kwargs)
    builder = lambda: MSTANet(**kwargs)
    curriculum = CurriculumSchedule(alpha_max=alpha_max) if use_curr else None
    res = run_chronological_cv(
        builder, samples,
        device=device, max_epochs=max_epochs,
        curriculum=curriculum, use_curriculum=use_curr,
        alpha_max=alpha_max,
    )
    return res["R2"]["mean"]


def shapley_values(samples,
                   base_kwargs: dict,
                   device: str = "cpu",
                   max_epochs: int = 30,
                   subsample_balanced: int = 20) -> Dict:
    """Compute Shapley values over the 32 ablation coalitions.

    `subsample_balanced` matches the paper's "20-game balanced sub-sample"
    of the longitudinal dataset; for full data, this argument can be ignored.
    """
    n = len(COMPONENTS)
    all_coalitions = list(chain.from_iterable(
        combinations(COMPONENTS, r) for r in range(n + 1)))

    v: Dict[frozenset, float] = {}
    for coal in all_coalitions:
        v[frozenset(coal)] = evaluate_coalition(
            list(coal), samples, base_kwargs,
            device=device, max_epochs=max_epochs,
        )

    # Shapley values
    phi = {comp: 0.0 for comp in COMPONENTS}
    for comp in COMPONENTS:
        others = [c for c in COMPONENTS if c != comp]
        for r in range(len(others) + 1):
            for subset in combinations(others, r):
                w = factorial(r) * factorial(n - r - 1) / factorial(n)
                S = frozenset(subset)
                S_with = frozenset(subset + (comp,))
                phi[comp] += w * (v[S_with] - v[S])

    # Pairwise interactions
    interactions = {}
    empty = v[frozenset()]
    for i, j in combinations(COMPONENTS, 2):
        v_i = v[frozenset([i])]
        v_j = v[frozenset([j])]
        v_ij = v[frozenset([i, j])]
        interactions[(i, j)] = 0.5 * (v_ij - v_i - v_j + empty)

    # Total R² improvement attributed to all components
    full_value = v[frozenset(COMPONENTS)]
    baseline_value = empty
    total_delta = full_value - baseline_value
    contributions_pct = {comp: 100.0 * phi[comp] / total_delta if total_delta else 0.0
                         for comp in COMPONENTS}
    interactions_pct = {
        f"{i} × {j}": 100.0 * interactions[(i, j)] / total_delta if total_delta else 0.0
        for (i, j) in interactions
    }
    return {
        "v_full": full_value,
        "v_empty": baseline_value,
        "total_delta_R2": total_delta,
        "phi": phi,
        "contributions_pct": contributions_pct,
        "interactions": interactions,
        "interactions_pct": interactions_pct,
        "coalition_values": {",".join(sorted(k)): val for k, val in v.items()},
    }
