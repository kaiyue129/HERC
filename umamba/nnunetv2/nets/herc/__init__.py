"""HERC: Heterogeneous Error Routing and Correction.

A shared correction cell that is attached to a frozen segmenter.  Residual errors
are typed into false-negative recovery, false-positive suppression, hierarchy
correction and boundary correction; a diagnostic head allocates the four actions
and a bounded logit update is applied and re-diagnosed at the next step.
"""

from .cell import HERCCell, HERCCorrector
from .config import HERCConfig
from .diagnosis import HERCDiagnosis
from .error_augmentation import inject_typed_error
from .experts import (
    BoundaryCorrectionExpert,
    FalseNegativeRecoveryExpert,
    FalsePositiveSuppressionExpert,
    ForegroundConfusionExpert,
    HierarchyCorrectionExpert,
)
from .targets import build_route_targets

__all__ = [
    'HERCConfig',
    'HERCCell',
    'HERCCorrector',
    'HERCDiagnosis',
    'FalseNegativeRecoveryExpert',
    'FalsePositiveSuppressionExpert',
    'ForegroundConfusionExpert',
    'HierarchyCorrectionExpert',
    'BoundaryCorrectionExpert',
    'inject_typed_error',
    'build_route_targets',
]
