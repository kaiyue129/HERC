"""Configuration of the shared HERC correction cell.

HERC (Heterogeneous Error Routing and Correction) attaches one shared correction
cell to a frozen segmenter.  The defaults reproduce the configuration reported in
Section 3.2 of the manuscript:

* three nested BraTS regions (WT, TC, ET) predicted by independent sigmoid heads,
* a 64^3 diagnostic stage and a 128^3 boundary stage,
* a semantic bound b_s = 2.0, a boundary bound b_b = 1.5 and a trust-region
  bound b_g = 2.0 (Equation 4),
* a step size ``eta`` in [0.05, 1.0],
* two correction steps for training and at most three at inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

#: Route order shared by the diagnosis, the route targets and the evaluator.
#: It follows Figure 3: false-negative recovery, false-positive suppression,
#: hierarchy correction, boundary correction.
ROUTE_NAMES: Tuple[str, ...] = ('fn', 'fp', 'hierarchy', 'boundary')

#: Region order of the BraTS region heads, as used throughout the manuscript.
REGION_NAMES: Tuple[str, ...] = ('WT', 'TC', 'ET')


@dataclass
class HERCConfig:
    """Static and iterative settings of one HERC correction cell.

    Attributes:
        num_outputs: Number of region heads produced by the frozen segmenter.
        has_regions: ``True`` for the region-based (sigmoid) BraTS contract,
            ``False`` for the atomic (softmax) multiclass fallback.
        wt_index: Head index of the whole-tumour region.
        tc_index: Head index of the tumour-core region.
        et_index: Head index of the enhancing-tumour region.
        diagnosis_feature_channels: Width of the coarse feature projection.
        diagnosis_hidden_channels: Width of the diagnostic context stem.
        coarse_checkpoint: Activation checkpointing at the coarse resolution.
        expert_hidden_channels: Width of the three semantic experts.
        semantic_residual_bound: ``b_s`` of Equation 3.
        boundary_feature_channels: Width of the full-resolution projection.
        boundary_hidden_channels: Width of the boundary context stem.
        boundary_residual_bound: ``b_b`` of the trust-region transform.
        boundary_radius: Radius of the soft morphological band.
        boundary_checkpoint: Activation checkpointing of the boundary expert.
        train_steps: Number of steps used during training.
        max_inference_steps: Upper bound on the inference iteration count.
        adaptive_stopping: Stop early once the predicted severity is small.
        stopping_threshold: Severity threshold of the stopping rule.
        stopping_topk_fraction: Top-k fraction of voxels used by that rule.
        min_step_size: Lower bound of ``eta``.
        max_step_size: Upper bound of ``eta``.
        global_update_bound: ``b_g`` of the trust-region transform.
        eps: Numerical guard.
        align_corners: Trilinear interpolation convention.
    """

    num_outputs: int = 3
    has_regions: bool = True
    wt_index: int = 0
    tc_index: int = 1
    et_index: int = 2

    diagnosis_feature_channels: int = 32
    diagnosis_hidden_channels: int = 32
    coarse_checkpoint: bool = True

    expert_hidden_channels: int = 24
    semantic_residual_bound: float = 2.0

    boundary_feature_channels: int = 8
    boundary_hidden_channels: int = 8
    boundary_residual_bound: float = 1.5
    boundary_radius: int = 1
    boundary_checkpoint: bool = True

    train_steps: int = 2
    max_inference_steps: int = 3
    adaptive_stopping: bool = True
    stopping_threshold: float = 0.025
    stopping_topk_fraction: float = 0.02
    min_step_size: float = 0.05
    max_step_size: float = 1.0
    global_update_bound: float = 2.0

    eps: float = 1e-6
    align_corners: bool = False

    def validate(self) -> None:
        """Check the settings against the region contract of the cell."""
        if self.num_outputs < 1:
            raise ValueError('num_outputs must be positive')
        if self.has_regions:
            if self.num_outputs != 3:
                raise ValueError(
                    'Region-aware HERC expects exactly three nnU-Net region heads '
                    '(WT, TC, ET).'
                )
            ids = (self.wt_index, self.tc_index, self.et_index)
            if sorted(ids) != [0, 1, 2]:
                raise ValueError('WT/TC/ET indices must be a permutation of (0, 1, 2).')
        if not 0.0 <= self.min_step_size <= self.max_step_size <= 1.0:
            raise ValueError('The step size must satisfy 0 <= min_step_size <= '
                             'max_step_size <= 1.')
        if self.train_steps < 1 or self.max_inference_steps < 1:
            raise ValueError('train_steps and max_inference_steps must be positive')
