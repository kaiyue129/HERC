"""Typed-error augmentation used by the warm-up phase of HERC.

The warm-up teaches the corrective operators on explicit, typed errors: a missed
region (FN), a spurious region (FP), a nesting violation and a boundary
displacement.  These are error-space perturbations of the logits, not random
image noise, so every step targets one of the four actions of Figure 1.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

#: Error types available in the warm-up phase, in route order.
ERROR_TYPES = ('fn', 'fp', 'hierarchy', 'boundary')

#: Error types of the atomic fallback, where nesting is replaced by confusion.
ATOMIC_ERROR_TYPES = ('fn', 'fp', 'confusion', 'boundary')


def _resize_region_target(target: torch.Tensor, spatial) -> torch.Tensor:
    if target.ndim != 5:
        raise ValueError('Region target must be [B, R, D, H, W]')
    if target.shape[2:] != spatial:
        target = F.interpolate(target.float(), size=spatial, mode='nearest')
    return target.float()


def _resize_atomic_target(target: torch.Tensor, spatial) -> torch.Tensor:
    if target.ndim == 5 and target.shape[1] == 1:
        target = target[:, 0]
    if target.shape[1:] != spatial:
        target = F.interpolate(target[:, None].float(), size=spatial, mode='nearest')[:, 0]
    return target.long()


@torch.no_grad()
def inject_region_error(
    logits: torch.Tensor,
    target: torch.Tensor,
    mode: str | None = None,
    wt: int = 0,
    tc: int = 1,
    et: int = 2,
):
    """Perturb region logits with one typed error.

    Args:
        logits: Region logits of the frozen segmenter.
        target: Region reference used to place the perturbation.
        mode: One of ``ERROR_TYPES``; sampled when ``None``.
        wt: Head index of WT.
        tc: Head index of TC.
        et: Head index of ET.

    Returns:
        ``(perturbed_logits, mode)``.
    """
    if mode is None:
        mode = random.choice(ERROR_TYPES)
    out = logits.detach().clone()
    y = _resize_region_target(target, out.shape[2:])
    strength = random.uniform(1.0, 2.5)

    if mode == 'fn':
        c = random.randrange(out.shape[1])
        mask = y[:, c:c + 1]
        eroded = -F.max_pool3d(-mask, 3, stride=1, padding=1)
        removed = (mask - eroded).clamp_min(0)
        if random.random() < 0.4:
            removed = torch.maximum(removed, mask * (torch.rand_like(mask) > 0.85))
        out[:, c:c + 1] -= strength * removed
    elif mode == 'fp':
        c = random.randrange(out.shape[1])
        mask = y[:, c:c + 1]
        dilated = F.max_pool3d(mask, 5, stride=1, padding=2)
        out[:, c:c + 1] += strength * (dilated - mask).clamp_min(0)
    elif mode == 'hierarchy':
        if random.random() < 0.5:
            mask = y[:, tc:tc + 1]
            out[:, tc:tc + 1] += strength * mask
            out[:, wt:wt + 1] -= 0.5 * strength * mask
        else:
            mask = y[:, et:et + 1]
            out[:, et:et + 1] += strength * mask
            out[:, tc:tc + 1] -= 0.5 * strength * mask
    elif mode == 'boundary':
        c = random.randrange(out.shape[1])
        mask = y[:, c:c + 1]
        if random.random() < 0.5:
            altered = -F.max_pool3d(-mask, 3, stride=1, padding=1)
            out[:, c:c + 1] -= strength * (mask - altered).clamp_min(0)
        else:
            altered = F.max_pool3d(mask, 3, stride=1, padding=1)
            out[:, c:c + 1] += strength * (altered - mask).clamp_min(0)
    else:
        raise ValueError(f'Unknown region error type: {mode}')
    return out, mode


@torch.no_grad()
def inject_atomic_error(
    logits: torch.Tensor,
    target: torch.Tensor,
    mode: str | None = None,
):
    """Perturb atomic logits with one typed error (nesting replaced by confusion)."""
    if mode is None:
        mode = random.choice(ATOMIC_ERROR_TYPES)
    out = logits.detach().clone()
    y = _resize_atomic_target(target, out.shape[2:])
    strength = random.uniform(1.0, 2.5)
    foreground = (y > 0).float()[:, None]

    if mode == 'fn':
        eroded = -F.max_pool3d(-foreground, 3, stride=1, padding=1)
        band = (foreground - eroded).clamp_min(0)
        out[:, 1:] -= strength * band
        out[:, :1] += strength * band
    elif mode == 'fp':
        dilated = F.max_pool3d(foreground, 5, stride=1, padding=2)
        band = (dilated - foreground).clamp_min(0)
        out[:, 1:] += strength * band
        out[:, :1] -= strength * band
    elif mode == 'confusion':
        a, b = random.sample(range(1, out.shape[1]), 2)
        region = (y == a).float()[:, None]
        out[:, a:a + 1] -= strength * region
        out[:, b:b + 1] += strength * region
    elif mode == 'boundary':
        return inject_atomic_error(logits, target, random.choice(('fn', 'fp')))
    else:
        raise ValueError(f'Unknown atomic error type: {mode}')
    return out, mode


def inject_typed_error(
    logits: torch.Tensor,
    target: torch.Tensor,
    has_regions: bool,
    mode: str | None = None,
    **kwargs,
):
    """Dispatch typed-error injection to the region or the atomic contract."""
    if has_regions:
        return inject_region_error(logits, target, mode=mode, **kwargs)
    return inject_atomic_error(logits, target, mode=mode)
