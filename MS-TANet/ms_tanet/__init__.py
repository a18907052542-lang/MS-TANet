"""
MS-TANet: Multi-Scale Temporal Attention Network for dynamic correlation modeling
of athletes' training load and sports performance.
"""

from ms_tanet.ms_dcc import MultiScaleDilatedCausalConv
from ms_tanet.sd_sa import SparseDecaySelfAttention
from ms_tanet.ie_fm import IndividualEmbeddingFeatureModulation
from ms_tanet.model import MSTANet
from ms_tanet.physics_loss import PhysicsConstrainedLoss

__all__ = [
    "MultiScaleDilatedCausalConv",
    "SparseDecaySelfAttention",
    "IndividualEmbeddingFeatureModulation",
    "MSTANet",
    "PhysicsConstrainedLoss",
]
