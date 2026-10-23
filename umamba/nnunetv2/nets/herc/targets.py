"""Supervision targets of the HERC correction cell.

Section 3.2 of the manuscript states that supervision is generated from the
current prediction and the reference at every step.  The severity target is the
maximum absolute region error, and the unnormalised route targets are the
per-region false-negative and false-positive evidence, the hierarchy violation
and the boundary-weighted error.  All four scores are normalised into a soft
routing distribution, so the routes are not mutually exclusive error labels.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .cues import hierarchy_violation_cue


def highest_target(target):
    """Return the full-resolution reference of a (possibly deep-supervised) target."""
    return target[0] if isinstance(target, (list, tuple)) else target


def resize_region_target(target: torch.Tensor, spatial) -> torch.Tensor:
    """Resample a ``[B, R, D, H, W]`` region reference to ``spatial``."""
    if target.ndim != 5:
        raise ValueError(f'Region target must be [B, R, D, H, W], got {tuple(target.shape)}')
    if target.shape[2:] != tuple(spatial):
        target = F.interpolate(target.float(), size=spatial, mode='nearest')
    return target.float()


def resize_atomic_target(target: torch.Tensor, spatial) -> torch.Tensor:
    """Resample an atomic ``[B, D, H, W]`` reference to ``spatial``."""
    if target.ndim == 5 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 4:
        raise ValueError(f'Atomic target must be [B, D, H, W], got {tuple(target.shape)}')
    if target.shape[1:] != tuple(spatial):
        target = F.interpolate(target[:, None].float(), size=spatial, mode='nearest')[:, 0]
    return target.long()


def region_boundary_mask(target: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Hard boundary band of a region reference (dilation minus erosion > 0)."""
    target = target.float()
    k = 2 * radius + 1
    dilation = F.max_pool3d(target, k, stride=1, padding=radius)
    erosion = -F.max_pool3d(-target, k, stride=1, padding=radius)
    return (dilation - erosion > 0).float()


def atomic_boundary_mask(target: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Hard boundary band of an atomic reference."""
    if target.ndim == 5 and target.shape[1] == 1:
        target = target[:, 0]
    t = target.float()[:, None]
    k = 2 * radius + 1
    maximum = F.max_pool3d(t, k, stride=1, padding=radius)
    minimum = -F.max_pool3d(-t, k, stride=1, padding=radius)
    return (maximum != minimum).float()


def build_region_route_targets(
    logits: torch.Tensor,
    target: torch.Tensor,
    wt: int = 0,
    tc: int = 1,
    et: int = 2,
    gamma: float = 1.0,
    boundary_radius: int = 1,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Build severity and route targets for the region (sigmoid) contract."""
    spatial = logits.shape[2:]
    target = resize_region_target(target, spatial)
    prob = torch.sigmoid(logits)
    abs_err = (prob - target).abs()

    severity = abs_err.max(dim=1, keepdim=True).values.pow(gamma)
    fn = (target * (1.0 - prob)).max(dim=1, keepdim=True).values
    fp = ((1.0 - target) * prob).max(dim=1, keepdim=True).values
    hierarchy = hierarchy_violation_cue(prob, wt=wt, tc=tc, et=et)

    band_per_region = region_boundary_mask(target, boundary_radius)
    boundary = (band_per_region * abs_err).max(dim=1, keepdim=True).values

    route = _normalise_route(torch.cat([fn, fp, hierarchy, boundary], dim=1), eps)

    return {
        'severity': severity,
        'route': route,
        'boundary_per_region': band_per_region,
        'hierarchy_violation': hierarchy,
        'prob': prob,
        'target': target,
    }


def build_atomic_route_targets(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 1.0,
    boundary_radius: int = 1,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Build severity and route targets for the atomic (softmax) contract."""
    spatial = logits.shape[2:]
    target = resize_atomic_target(target, spatial)
    prob = torch.softmax(logits, dim=1)
    p_gt = prob.gather(1, target[:, None])
    severity = (1.0 - p_gt).clamp(0, 1).pow(gamma)

    gt_bg = (target == 0).float()[:, None]
    gt_fg = 1.0 - gt_bg
    p_bg = prob[:, :1]
    fn = gt_fg * p_bg
    fp = gt_bg * (1.0 - p_bg)

    onehot = F.one_hot(target, num_classes=prob.shape[1]).movedim(-1, 1).float()
    correct = (prob * onehot).sum(dim=1, keepdim=True)
    foreground_prob = prob[:, 1:].sum(dim=1, keepdim=True)
    confusion = gt_fg * (foreground_prob - correct).clamp_min(0)

    band = atomic_boundary_mask(target, boundary_radius)
    boundary = band * (1.0 - p_gt)

    route = _normalise_route(torch.cat([fn, fp, confusion, boundary], dim=1), eps)

    return {
        'severity': severity,
        'route': route,
        'boundary': band,
        'prob': prob,
        'target': target,
    }


def build_route_targets(
    logits: torch.Tensor,
    target: torch.Tensor,
    has_regions: bool,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Dispatch target construction to the region or the atomic builder."""
    if has_regions:
        return build_region_route_targets(logits, target, **kwargs)
    return build_atomic_route_targets(logits, target, **kwargs)


def _normalise_route(scores: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalise route scores, falling back to a uniform route where they vanish."""
    denominator = scores.sum(dim=1, keepdim=True)
    route = scores / denominator.clamp_min(eps)
    uniform = torch.full_like(route, 1.0 / route.shape[1])
    return torch.where(denominator <= eps, uniform, route)


__all__ = [
    'highest_target',
    'resize_region_target',
    'resize_atomic_target',
    'region_boundary_mask',
    'atomic_boundary_mask',
    'build_region_route_targets',
    'build_atomic_route_targets',
    'build_route_targets',
]
