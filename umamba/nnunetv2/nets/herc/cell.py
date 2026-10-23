"""The shared HERC correction cell and its iterative wrapper.

Section 3.2 of the manuscript: "HERC applies one shared correction cell twice to
the current logits."  One cell performs diagnose -> typed experts -> bounded
update; :class:`HERCCorrector` applies the same cell for ``T`` steps, re-diagnosing
the updated state before every step after the first.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import HERCConfig
from .diagnosis import HERCDiagnosis
from .experts import (
    BoundaryCorrectionExpert,
    FalseNegativeRecoveryExpert,
    FalsePositiveSuppressionExpert,
    ForegroundConfusionExpert,
    HierarchyCorrectionExpert,
)


def expand_foreground_to_full(residual: torch.Tensor, num_outputs: int) -> torch.Tensor:
    """Prepend a zero background head to an atomic foreground residual."""
    return torch.cat([torch.zeros_like(residual[:, :1]), residual], dim=1)


class HERCCell(nn.Module):
    """One diagnose-and-correct step: (logits, features) -> updated logits.

    Args:
        coarse_decoder_channels: Channels of the penultimate decoder feature.
        full_decoder_channels: Channels of the final decoder feature.
        cfg: Cell configuration.
    """

    def __init__(
        self,
        coarse_decoder_channels: int,
        full_decoder_channels: int,
        cfg: HERCConfig,
    ):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.diagnosis = HERCDiagnosis(
            coarse_decoder_channels,
            cfg.num_outputs,
            cfg.has_regions,
            cfg.diagnosis_feature_channels,
            cfg.diagnosis_hidden_channels,
            cfg.wt_index,
            cfg.tc_index,
            cfg.et_index,
            cfg.coarse_checkpoint,
        )
        expert_in = cfg.diagnosis_hidden_channels + cfg.num_outputs
        corrected_channels = cfg.num_outputs if cfg.has_regions else cfg.num_outputs - 1

        self.fn_expert = FalseNegativeRecoveryExpert(
            expert_in,
            cfg.expert_hidden_channels,
            corrected_channels,
            cfg.semantic_residual_bound,
            cfg.coarse_checkpoint,
        )
        self.fp_expert = FalsePositiveSuppressionExpert(
            expert_in,
            cfg.expert_hidden_channels,
            corrected_channels,
            cfg.semantic_residual_bound,
            cfg.coarse_checkpoint,
        )
        if cfg.has_regions:
            self.hierarchy_expert = HierarchyCorrectionExpert(
                expert_in,
                cfg.expert_hidden_channels,
                cfg.wt_index,
                cfg.tc_index,
                cfg.et_index,
                cfg.semantic_residual_bound,
                cfg.coarse_checkpoint,
            )
        else:
            self.hierarchy_expert = ForegroundConfusionExpert(
                expert_in,
                cfg.expert_hidden_channels,
                corrected_channels,
                cfg.semantic_residual_bound,
                cfg.coarse_checkpoint,
            )

        self.boundary_expert = BoundaryCorrectionExpert(
            full_decoder_channels,
            cfg.num_outputs,
            cfg.has_regions,
            cfg.boundary_feature_channels,
            cfg.boundary_hidden_channels,
            cfg.boundary_residual_bound,
            cfg.boundary_radius,
            cfg.boundary_checkpoint,
        )

    def prepare_static_features(
        self,
        image: torch.Tensor,
        coarse_feature: torch.Tensor,
        full_feature: torch.Tensor,
    ) -> dict:
        """Compute the projections and gradients that do not change across steps.

        The backbone features and the image are constant over the correction
        iterations, so projecting them once removes redundant work.
        """
        return {
            'coarse_projected': self.diagnosis.project_feature(coarse_feature),
            'full_projected': self.boundary_expert.project_feature(full_feature),
            'image_grad': self.boundary_expert.prepare_image_gradient(image),
        }

    def forward(
        self,
        image: torch.Tensor,
        coarse_feature: torch.Tensor,
        full_feature: torch.Tensor,
        current_logits: torch.Tensor,
        static: Optional[dict] = None,
    ) -> dict:
        """Apply one correction step.

        Returns:
            ``{'logits': updated_logits, 'aux': {...}}`` where ``aux`` holds the
            diagnosis outputs, the individual expert residuals, the fused
            residual and the applied step.
        """
        if static is None:
            static = self.prepare_static_features(image, coarse_feature, full_feature)

        diagnosis = self.diagnosis(
            coarse_feature,
            current_logits,
            projected_feature=static['coarse_projected'],
        )
        context = torch.cat([diagnosis['context'], diagnosis['prob_coarse']], dim=1)
        w_fn, w_fp, w_h, w_b = [diagnosis['route_weights'][:, i:i + 1] for i in range(4)]

        r_fn = self.fn_expert(context)
        r_fp = self.fp_expert(context)
        r_h = (
            self.hierarchy_expert(context, diagnosis['prob_coarse'])
            if self.cfg.has_regions
            else self.hierarchy_expert(context)
        )

        # Equation 3: severity-weighted sum of the three semantic residuals.
        residual_low = diagnosis['severity'] * (w_fn * r_fn + w_fp * r_fp + w_h * r_h)
        if not self.cfg.has_regions:
            residual_low = expand_foreground_to_full(residual_low, self.cfg.num_outputs)

        residual_semantic = F.interpolate(
            residual_low,
            size=current_logits.shape[2:],
            mode='trilinear',
            align_corners=self.cfg.align_corners,
        )

        residual_boundary, boundary_aux = self.boundary_expert(
            full_feature,
            current_logits,
            w_b,
            image,
            projected_feature=static['full_projected'],
            image_grad=static['image_grad'],
        )
        if not self.cfg.has_regions:
            residual_boundary = expand_foreground_to_full(
                residual_boundary, self.cfg.num_outputs
            )
        if residual_boundary.shape[2:] != current_logits.shape[2:]:
            residual_boundary = F.interpolate(
                residual_boundary,
                size=current_logits.shape[2:],
                mode='trilinear',
                align_corners=self.cfg.align_corners,
            )

        # Equation 4: trust-region transform, step size, state update.
        residual = residual_semantic + residual_boundary
        residual = self.cfg.global_update_bound * torch.tanh(
            residual / max(self.cfg.global_update_bound, self.cfg.eps)
        )

        step = torch.sigmoid(diagnosis['step_logit'])
        eta = self.cfg.min_step_size + (
            self.cfg.max_step_size - self.cfg.min_step_size
        ) * step
        delta = eta[:, :, None, None, None] * residual
        updated = current_logits + delta

        aux = dict(diagnosis)
        aux.update({
            'r_fn': r_fn,
            'r_fp': r_fp,
            'r_h': r_h,
            'r_semantic': residual_semantic,
            'r_boundary': residual_boundary,
            'delta': delta,
            'eta': eta,
            **boundary_aux,
        })
        return {'logits': updated, 'aux': aux}


class HERCCorrector(nn.Module):
    """Iterative wrapper: apply one shared :class:`HERCCell` for ``T`` steps.

    Training uses ``cfg.train_steps`` steps; inference uses at most
    ``cfg.max_inference_steps`` steps and may stop early when the predicted
    severity falls below ``cfg.stopping_threshold``.
    """

    def __init__(
        self,
        coarse_decoder_channels: int,
        full_decoder_channels: int,
        cfg: HERCConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.cell = HERCCell(coarse_decoder_channels, full_decoder_channels, cfg)

    def forward(
        self,
        image: torch.Tensor,
        coarse_feature: torch.Tensor,
        full_feature: torch.Tensor,
        initial_logits: torch.Tensor,
        steps: Optional[int] = None,
    ) -> dict:
        """Run the correction loop.

        Args:
            image: Input patch.
            coarse_feature: Penultimate decoder feature of the frozen backbone.
            full_feature: Final decoder feature of the frozen backbone.
            initial_logits: Logits produced by the frozen segmenter.
            steps: Overrides the configured number of steps.

        Returns:
            ``{'initial_logits', 'final_logits', 'step_states'}``.
        """
        logits = initial_logits
        states: List[dict] = []
        static = self.cell.prepare_static_features(image, coarse_feature, full_feature)
        if steps is None:
            steps = self.cfg.train_steps if self.training else self.cfg.max_inference_steps

        active = torch.ones(
            logits.shape[0],
            1,
            1,
            1,
            1,
            device=logits.device,
            dtype=logits.dtype,
        )

        for _ in range(int(steps)):
            out = self.cell(image, coarse_feature, full_feature, logits, static=static)
            proposed = out['logits']
            if (not self.training) and self.cfg.adaptive_stopping:
                severity = out['aux']['severity'].flatten(1)
                fraction = min(
                    max(float(self.cfg.stopping_topk_fraction), 1.0 / severity.shape[1]),
                    1.0,
                )
                k = max(1, int(round(fraction * severity.shape[1])))
                error_mass = torch.topk(severity, k=k, dim=1).values.mean(dim=1, keepdim=True)
                error_mass = error_mass[:, :, None, None, None]
                still_active = (error_mass > self.cfg.stopping_threshold).to(logits.dtype)
                active = active * still_active
                logits = active * proposed + (1.0 - active) * logits
                out['aux']['error_mass'] = error_mass
                out['aux']['active_mask'] = active
            else:
                logits = proposed
            states.append(out['aux'])
            if (not self.training) and self.cfg.adaptive_stopping and active.max().item() == 0:
                break

        return {
            'initial_logits': initial_logits,
            'final_logits': logits,
            'step_states': states,
        }
