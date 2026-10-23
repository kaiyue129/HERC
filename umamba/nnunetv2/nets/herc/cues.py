"""Diagnostic cues and morphological operators of the HERC correction cell.

Section 3.2 of the manuscript defines four cues for the coarse state of the
prediction: the mean normalised Bernoulli entropy, the threshold ambiguity, the
magnitude of the probability gradient and the hierarchy cue of Equation 1.  The
same module hosts the soft morphological band used by the boundary expert.

Every function is differentiable and free of learned parameters, so the cues can
be recomputed at each correction step at negligible cost.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def probabilities_from_logits(logits: torch.Tensor, has_regions: bool) -> torch.Tensor:
    """Map logits to probabilities under the region (sigmoid) or atomic (softmax) contract."""
    return torch.sigmoid(logits) if has_regions else torch.softmax(logits, dim=1)


def entropy_cue(prob: torch.Tensor, has_regions: bool, eps: float = 1e-6) -> torch.Tensor:
    """Mean normalised Bernoulli entropy of the region heads (first cue).

    The region contract averages the normalised binary entropy over WT, TC and ET;
    the atomic contract uses the normalised categorical entropy instead.
    """
    p = prob.clamp(eps, 1 - eps)
    if has_regions:
        entropy = -(p * p.log() + (1 - p) * (1 - p).log())
        return entropy.mean(dim=1, keepdim=True) / 0.6931471805599453
    entropy = -(p * p.log()).sum(dim=1, keepdim=True)
    norm = torch.log(torch.tensor(prob.shape[1], device=prob.device, dtype=prob.dtype))
    return entropy / norm.clamp_min(eps)


def ambiguity_cue(prob: torch.Tensor, has_regions: bool) -> torch.Tensor:
    """Threshold ambiguity ``max_r (1 - 2 |p_r - 0.5|)`` (second cue)."""
    if has_regions:
        return (1.0 - 2.0 * (prob - 0.5).abs()).clamp(0, 1).max(dim=1, keepdim=True).values
    top2 = torch.topk(prob, k=min(2, prob.shape[1]), dim=1).values
    if top2.shape[1] == 1:
        return 1.0 - top2
    return (1.0 - (top2[:, :1] - top2[:, 1:2])).clamp(0, 1)


def probability_gradient_cue(prob: torch.Tensor) -> torch.Tensor:
    """Magnitude of the probability gradient along the three spatial axes (third cue)."""
    dz = F.pad(prob[:, :, 1:] - prob[:, :, :-1], (0, 0, 0, 0, 0, 1))
    dy = F.pad(prob[:, :, :, 1:] - prob[:, :, :, :-1], (0, 0, 0, 1, 0, 0))
    dx = F.pad(prob[:, :, :, :, 1:] - prob[:, :, :, :, :-1], (0, 1, 0, 0, 0, 0))
    magnitude = torch.sqrt(dx.square() + dy.square() + dz.square() + 1e-8)
    return magnitude.mean(dim=1, keepdim=True)


def image_gradient_cue(image: torch.Tensor) -> torch.Tensor:
    """Gradient magnitude of the input image, fed to the boundary expert."""
    return probability_gradient_cue(image)


def hierarchy_violation_cue(
    prob: torch.Tensor,
    wt: int = 0,
    tc: int = 1,
    et: int = 2,
) -> torch.Tensor:
    """Hierarchy cue ``h_t`` of Equation 1: ``ReLU(p_TC - p_WT) + ReLU(p_ET - p_TC)``."""
    v_tc = F.relu(prob[:, tc:tc + 1] - prob[:, wt:wt + 1])
    v_et = F.relu(prob[:, et:et + 1] - prob[:, tc:tc + 1])
    return v_tc + v_et


def soft_morphological_band(prob: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Soft morphological band: max-pooled dilation minus min-pooled erosion.

    This reproduces the "radius one" band of Section 3.2 without thresholding the
    probability map, so the residual can be differentiated through it.
    """
    k = 2 * radius + 1
    dilation = F.max_pool3d(prob, k, stride=1, padding=radius)
    erosion = -F.max_pool3d(-prob, k, stride=1, padding=radius)
    return (dilation - erosion).clamp(0, 1)
