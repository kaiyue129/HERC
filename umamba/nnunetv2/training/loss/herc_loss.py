"""Objective of the HERC correction cell (Equation 5 of the manuscript).

The exact nnU-Net Dice + BCE/CE primitive (including deep supervision) remains the
main loss; the six auxiliary terms supervise the diagnosis and keep the iterative
correction safe:

``L = L_seg + lambda_s L_sev + lambda_r L_route + lambda_b L_bd
      + lambda_m L_mono + lambda_safe L_safe + lambda_h L_hier``

Two further terms, ``L_anchor`` and ``L_coarse``, are optional additions that are
disabled by default (``lambda_anchor = lambda_coarse = 0``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from nnunetv2.nets.herc.cues import hierarchy_violation_cue
from nnunetv2.nets.herc.targets import (
    atomic_boundary_mask,
    build_route_targets,
    highest_target,
    region_boundary_mask,
    resize_atomic_target,
    resize_region_target,
)


def highest_output(output):
    """Return the full-resolution tensor of a (possibly deep-supervised) output."""
    return output[0] if isinstance(output, (list, tuple)) else output


def region_aux_seg_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-5):
    """Dice + BCE primitive for the region (sigmoid) contract."""
    target = resize_region_target(target, logits.shape[2:])
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    dims = tuple(range(2, prob.ndim))
    intersection = (prob * target).sum(dims)
    denominator = prob.sum(dims) + target.sum(dims)
    dice = 1.0 - ((2 * intersection + eps) / (denominator + eps)).mean()
    return bce + dice


def atomic_aux_seg_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-5):
    """Dice + CE primitive for the atomic (softmax) contract."""
    target = resize_atomic_target(target, logits.shape[2:])
    ce = F.cross_entropy(logits, target)
    prob = torch.softmax(logits, dim=1)
    onehot = F.one_hot(target, num_classes=logits.shape[1]).movedim(-1, 1).float()
    # nnU-Net's standard atomic Dice excludes background.
    prob_fg = prob[:, 1:]
    onehot_fg = onehot[:, 1:]
    dims = tuple(range(2, prob.ndim))
    intersection = (prob_fg * onehot_fg).sum(dims)
    denominator = prob_fg.sum(dims) + onehot_fg.sum(dims)
    dice = 1.0 - ((2 * intersection + eps) / (denominator + eps)).mean()
    return ce + dice


def aux_seg_loss(logits: torch.Tensor, target: torch.Tensor, has_regions: bool):
    """Segmentation primitive of the corrected state."""
    return region_aux_seg_loss(logits, target) if has_regions else atomic_aux_seg_loss(
        logits, target
    )


def boundary_loss(logits: torch.Tensor, target: torch.Tensor, has_regions: bool, radius: int = 1):
    """Boundary-weighted error ``L_bd``, restricted to the reference band."""
    if has_regions:
        target = resize_region_target(target, logits.shape[2:])
        band = region_boundary_mask(target, radius)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        return (bce * band).sum() / band.sum().clamp_min(1.0)
    target = resize_atomic_target(target, logits.shape[2:])
    band = atomic_boundary_mask(target, radius)[:, 0]
    ce = F.cross_entropy(logits, target, reduction='none')
    return (ce * band).sum() / band.sum().clamp_min(1.0)


class HERCCompositeLoss(nn.Module):
    """Main segmentation loss plus the five auxiliary HERC terms.

    Args:
        base_seg_loss: nnU-Net Dice + BCE/CE loss, possibly deep-supervised.
        has_regions: Whether the region-based BraTS contract is active.
        wt_index: Head index of WT.
        tc_index: Head index of TC.
        et_index: Head index of ET.
        lambda_severity: Weight of ``L_sev`` (smooth L1 on the severity map).
        lambda_route: Weight of ``L_route`` (severity-weighted soft cross-entropy).
        lambda_boundary: Weight of ``L_bd``.
        lambda_monotonic: Weight of ``L_mono``, which penalises an increase of the
            segmentation loss between adjacent steps.
        lambda_safety: Weight of ``L_safe``, which anchors the final loss to the
            initial prediction.
        lambda_hierarchy: Weight of ``L_hier``, the hierarchy violation rate.
        lambda_anchor: Weight of the joint-phase preservation anchor.
        lambda_coarse: Weight of the joint-phase coarse-branch supervision.
        boundary_radius: Radius of the reference boundary band.
    """

    def __init__(
        self,
        base_seg_loss,
        has_regions: bool,
        wt_index: int = 0,
        tc_index: int = 1,
        et_index: int = 2,
        lambda_severity: float = 0.20,
        lambda_route: float = 0.10,
        lambda_boundary: float = 0.10,
        lambda_monotonic: float = 0.20,
        lambda_safety: float = 0.10,
        lambda_hierarchy: float = 0.05,
        lambda_anchor: float = 0.0,
        lambda_coarse: float = 0.0,
        boundary_radius: int = 1,
    ):
        super().__init__()
        self.base_seg_loss = base_seg_loss
        # With deep supervision enabled, use the exact Dice + BCE/CE primitive for
        # the corrected full-resolution output.  Lower-resolution outputs are not
        # corrected and must not dilute the correction gradient.
        self.highres_seg_loss = getattr(base_seg_loss, 'loss', base_seg_loss)
        self.has_regions = has_regions
        self.wt, self.tc, self.et = wt_index, tc_index, et_index
        self.lambda_severity = lambda_severity
        self.lambda_route = lambda_route
        self.lambda_boundary = lambda_boundary
        self.lambda_monotonic = lambda_monotonic
        self.lambda_safety = lambda_safety
        self.lambda_hierarchy = lambda_hierarchy
        self.lambda_anchor = lambda_anchor
        self.lambda_coarse = lambda_coarse
        self.boundary_radius = boundary_radius
        self.last_components: dict = {}

    def forward(self, output, target):
        # Validation and inference may use the plain SSDMoE output contract.
        if not isinstance(output, dict):
            return self.base_seg_loss(output, target)

        corrected_outputs = output['seg_outputs']
        initial_outputs = output['initial_outputs']
        baseline_outputs = output['baseline_outputs']
        herc = output['herc']

        target_hi = highest_target(target)
        final_hi = highest_output(corrected_outputs)
        initial_hi = highest_output(initial_outputs)

        loss_final = self.highres_seg_loss(final_hi, target_hi)
        with torch.no_grad():
            loss_initial_main = self.highres_seg_loss(initial_hi, target_hi).detach()

        previous_logits = initial_hi
        previous_aux = self.highres_seg_loss(previous_logits, target_hi).detach()

        severity_losses = []
        route_losses = []
        boundary_losses = []
        monotonic_terms = []

        for state in herc['step_states']:
            spatial = state['severity'].shape[2:]
            diag_logits = F.interpolate(
                previous_logits,
                size=spatial,
                mode='trilinear',
                align_corners=False,
            )
            kwargs = {'boundary_radius': self.boundary_radius}
            if self.has_regions:
                kwargs.update({'wt': self.wt, 'tc': self.tc, 'et': self.et})
            route_target = build_route_targets(
                diag_logits.detach(), target_hi, self.has_regions, **kwargs
            )

            severity_losses.append(
                F.smooth_l1_loss(state['severity'], route_target['severity'])
            )

            log_route = state['route_logits'].log_softmax(dim=1)
            route_ce = -(route_target['route'] * log_route).sum(dim=1, keepdim=True)
            weight = route_target['severity'].detach()
            route_losses.append((route_ce * weight).sum() / weight.sum().clamp_min(1.0))

            previous_logits = previous_logits + state['delta']
            current_aux = self.highres_seg_loss(previous_logits, target_hi)
            monotonic_terms.append(F.relu(current_aux - previous_aux))
            previous_aux = current_aux.detach()
            boundary_losses.append(
                boundary_loss(previous_logits, target_hi, self.has_regions, self.boundary_radius)
            )

        zero = loss_final.new_zeros(())
        loss_severity = torch.stack(severity_losses).mean() if severity_losses else zero
        loss_route = torch.stack(route_losses).mean() if route_losses else zero
        loss_boundary = torch.stack(boundary_losses).mean() if boundary_losses else zero
        loss_monotonic = torch.stack(monotonic_terms).mean() if monotonic_terms else zero
        loss_safety = F.relu(loss_final - loss_initial_main)

        loss_hierarchy = zero
        if self.has_regions:
            final_prob = torch.sigmoid(highest_output(corrected_outputs))
            loss_hierarchy = hierarchy_violation_cue(final_prob, self.wt, self.tc, self.et).mean()

        # Joint-phase preservation anchor.  It is intentionally GT-weighted: it is
        # strongest where the coarse prediction already agrees with the reference
        # and weak where adaptation is needed.
        base_hi = highest_output(baseline_outputs)
        loss_anchor = zero
        if self.lambda_anchor > 0:
            kwargs = {'boundary_radius': self.boundary_radius}
            if self.has_regions:
                kwargs.update({'wt': self.wt, 'tc': self.tc, 'et': self.et})
            base_targets = build_route_targets(
                base_hi.detach(), target_hi, self.has_regions, **kwargs
            )
            preserve = (1.0 - base_targets['severity']).detach()
            if self.has_regions:
                p = torch.sigmoid(base_hi.detach()).clamp(1e-5, 1 - 1e-5)
                q = torch.sigmoid(final_hi).clamp(1e-5, 1 - 1e-5)
                kl = p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())
                kl = kl.mean(dim=1, keepdim=True)
            else:
                p = torch.softmax(base_hi.detach(), dim=1)
                log_q = torch.log_softmax(final_hi, dim=1)
                kl = (p * (p.clamp_min(1e-6).log() - log_q)).sum(dim=1, keepdim=True)
            loss_anchor = (kl * preserve).sum() / preserve.sum().clamp_min(1.0)

        # Keep the coarse branch itself competent during joint fine-tuning.
        loss_coarse = (
            self.base_seg_loss(baseline_outputs, target)
            if isinstance(baseline_outputs, (list, tuple))
            else self.highres_seg_loss(base_hi, target_hi)
        )

        total = (
            loss_final
            + self.lambda_severity * loss_severity
            + self.lambda_route * loss_route
            + self.lambda_boundary * loss_boundary
            + self.lambda_monotonic * loss_monotonic
            + self.lambda_safety * loss_safety
            + self.lambda_hierarchy * loss_hierarchy
            + self.lambda_anchor * loss_anchor
            + self.lambda_coarse * loss_coarse
        )

        self.last_components = {
            'total': total.detach(),
            'main': loss_final.detach(),
            'severity': loss_severity.detach(),
            'route': loss_route.detach(),
            'boundary': loss_boundary.detach(),
            'monotonic': loss_monotonic.detach(),
            'safety': loss_safety.detach(),
            'hierarchy': loss_hierarchy.detach(),
            'anchor': loss_anchor.detach(),
            'coarse': loss_coarse.detach(),
        }
        return total
