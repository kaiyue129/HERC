"""Diagnosis stage of the HERC correction cell (Section 3.2, Figure 3).

The diagnosis operates at the penultimate decoder resolution (64^3 for the BraTS
configuration).  It concatenates a 32-channel projection of the coarse feature
with the current region probabilities and the four diagnostic cues, converts them
through a context stem, and predicts three quantities:

* a severity map ``m_t`` that says where the prediction is unreliable,
* four route logits whose voxel-wise softmax gives the action weights ``w_t``,
* a spatially pooled step logit that gives the scalar step size ``eta_t``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .blocks import ContextStem, ConvNormAct
from .cues import (
    ambiguity_cue,
    entropy_cue,
    hierarchy_violation_cue,
    probabilities_from_logits,
    probability_gradient_cue,
)


class HERCDiagnosis(nn.Module):
    """Shared diagnostic head of the correction cell.

    Args:
        decoder_channels: Channels of the penultimate decoder feature.
        num_outputs: Number of region heads of the frozen segmenter.
        has_regions: Whether the region-based BraTS contract is active.
        feature_channels: Width of the coarse feature projection.
        hidden_channels: Width of the context stem.
        wt: Head index of WT.
        tc: Head index of TC.
        et: Head index of ET.
        use_checkpoint: Activation checkpointing for the context stem.

    Note:
        The route order is fixed to ``(fn, fp, hierarchy, boundary)`` in region
        mode and ``(fn, fp, confusion, boundary)`` in atomic mode, matching
        :data:`herc.config.ROUTE_NAMES`.
    """

    def __init__(
        self,
        decoder_channels: int,
        num_outputs: int,
        has_regions: bool,
        feature_channels: int = 32,
        hidden_channels: int = 32,
        wt: int = 0,
        tc: int = 1,
        et: int = 2,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.has_regions = has_regions
        self.wt, self.tc, self.et = wt, tc, et
        self.use_checkpoint = bool(use_checkpoint)

        self.feature_projection = ConvNormAct(decoder_channels, feature_channels, 1, 0)
        cue_channels = 4 if has_regions else 3
        self.stem = ContextStem(
            feature_channels + num_outputs + cue_channels,
            hidden_channels,
        )
        self.severity_head = nn.Conv3d(hidden_channels, 1, 1)
        self.route_head = nn.Conv3d(hidden_channels, 4, 1)
        self.step_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(hidden_channels, max(8, hidden_channels // 2), 1),
            nn.SiLU(inplace=True),
            nn.Conv3d(max(8, hidden_channels // 2), 1, 1),
        )

        # A low initial severity and a negative initial step logit keep every
        # correction close to zero at the start of training.
        nn.init.zeros_(self.severity_head.weight)
        nn.init.constant_(self.severity_head.bias, -2.5)
        nn.init.zeros_(self.route_head.weight)
        nn.init.zeros_(self.route_head.bias)
        nn.init.zeros_(self.step_head[-1].weight)
        nn.init.constant_(self.step_head[-1].bias, -1.0)

    def project_feature(self, decoder_feature: torch.Tensor) -> torch.Tensor:
        """Project the coarse decoder feature to ``feature_channels``."""
        return self.feature_projection(decoder_feature)

    def forward(
        self,
        decoder_feature: torch.Tensor,
        current_logits: torch.Tensor,
        projected_feature: torch.Tensor | None = None,
    ) -> dict:
        """Diagnose the current state and return its severity, routes and step size.

        Args:
            decoder_feature: Penultimate decoder feature.
            current_logits: Region logits of the current step.
            projected_feature: Cached projection of ``decoder_feature``.

        Returns:
            A dictionary with the context feature, the coarse probabilities, the
            severity map, the route logits and weights, the pooled step logit and
            the diagnostic cues.
        """
        spatial = decoder_feature.shape[2:]
        prob = probabilities_from_logits(current_logits, self.has_regions)
        prob_coarse = F.interpolate(prob, size=spatial, mode='trilinear', align_corners=False)

        entropy = entropy_cue(prob_coarse, self.has_regions)
        ambiguity = ambiguity_cue(prob_coarse, self.has_regions)
        gradient = probability_gradient_cue(prob_coarse)
        feature = (
            projected_feature
            if projected_feature is not None
            else self.project_feature(decoder_feature)
        )

        cues = [feature, prob_coarse, entropy, ambiguity, gradient]
        hierarchy = None
        if self.has_regions:
            hierarchy = hierarchy_violation_cue(prob_coarse, self.wt, self.tc, self.et)
            cues.append(hierarchy)

        stem_input = torch.cat(cues, dim=1)
        if self.training and self.use_checkpoint:
            context = checkpoint(self.stem, stem_input, use_reentrant=False)
        else:
            context = self.stem(stem_input)

        route_logits = self.route_head(context)
        out = {
            'context': context,
            'prob_coarse': prob_coarse,
            'severity': torch.sigmoid(self.severity_head(context)),
            'route_logits': route_logits,
            'route_weights': route_logits.softmax(dim=1),
            'step_logit': self.step_head(context).flatten(1),
            'entropy': entropy,
            'ambiguity': ambiguity,
            'prob_gradient': gradient,
        }
        if hierarchy is not None:
            out['hierarchy_violation'] = hierarchy
        return out
