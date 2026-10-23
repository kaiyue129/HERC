"""Typed experts of the HERC correction cell (Section 3.2, Figure 3).

Three semantic experts act at the coarse resolution and the boundary expert acts
at the full resolution.  The sign restrictions are properties of the individual
experts, not of their weighted sum: several routes may contribute at the same
voxel, so the total residual can take either sign.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .blocks import ContextStem, ConvNormAct, ResidualOperator
from .cues import image_gradient_cue, soft_morphological_band


class FalseNegativeRecoveryExpert(nn.Module):
    """Non-negative residual ``b_s * sigmoid(f_FN(.))`` for missed foreground."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        bound: float = 2.0,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.bound = float(bound)
        self.use_checkpoint = bool(use_checkpoint)
        self.net = ResidualOperator(
            in_channels, hidden_channels, out_channels, output_init='small_negative'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = (
            checkpoint(self.net, x, use_reentrant=False)
            if self.training and self.use_checkpoint
            else self.net(x)
        )
        return self.bound * torch.sigmoid(raw)


class FalsePositiveSuppressionExpert(nn.Module):
    """Non-positive residual ``-b_s * sigmoid(f_FP(.))`` for spurious foreground.

    This is the structural opposite of :class:`FalseNegativeRecoveryExpert` and
    shares its parameter budget.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        bound: float = 2.0,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.bound = float(bound)
        self.use_checkpoint = bool(use_checkpoint)
        self.net = ResidualOperator(
            in_channels, hidden_channels, out_channels, output_init='small_negative'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = (
            checkpoint(self.net, x, use_reentrant=False)
            if self.training and self.use_checkpoint
            else self.net(x)
        )
        return -self.bound * torch.sigmoid(raw)


class ForegroundConfusionExpert(nn.Module):
    """Atomic-mode replacement for the hierarchy expert.

    The atomic contract has no nesting relation, so the third route represents
    foreground class confusion instead.  The residual is centred across classes
    and therefore cannot change the foreground/background decision on its own.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        bound: float = 2.0,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.bound = float(bound)
        self.use_checkpoint = bool(use_checkpoint)
        self.net = ResidualOperator(in_channels, hidden_channels, out_channels, 'zero')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = (
            checkpoint(self.net, x, use_reentrant=False)
            if self.training and self.use_checkpoint
            else self.net(x)
        )
        residual = self.bound * torch.tanh(raw)
        return residual - residual.mean(dim=1, keepdim=True)


class HierarchyCorrectionExpert(nn.Module):
    """Hierarchy correction for the nested chain WT >= TC >= ET.

    The expert predicts four non-negative magnitudes (WT boost, TC suppress,
    TC boost, ET suppress) and multiplies them by the two violation terms of
    Equation 1 before writing into the three region heads.  It therefore acts
    only where the current prediction violates the nesting order.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        wt: int = 0,
        tc: int = 1,
        et: int = 2,
        bound: float = 1.5,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.wt, self.tc, self.et = wt, tc, et
        self.bound = float(bound)
        self.use_checkpoint = bool(use_checkpoint)
        # Output channels: WT boost, TC suppress, TC boost, ET suppress.
        self.net = ResidualOperator(in_channels, hidden_channels, 4, 'small_negative')

    def forward(self, x: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
        raw = (
            checkpoint(self.net, x, use_reentrant=False)
            if self.training and self.use_checkpoint
            else self.net(x)
        )
        actions = self.bound * torch.sigmoid(raw)
        a_wt_up, a_tc_down, a_tc_up, a_et_down = actions.chunk(4, dim=1)
        v_tc = F.relu(prob[:, self.tc:self.tc + 1] - prob[:, self.wt:self.wt + 1])
        v_et = F.relu(prob[:, self.et:self.et + 1] - prob[:, self.tc:self.tc + 1])

        delta = torch.zeros_like(prob)
        delta[:, self.wt:self.wt + 1] += v_tc * a_wt_up
        delta[:, self.tc:self.tc + 1] -= v_tc * a_tc_down
        delta[:, self.tc:self.tc + 1] += v_et * a_tc_up
        delta[:, self.et:self.et + 1] -= v_et * a_et_down
        return delta


class BoundaryCorrectionExpert(nn.Module):
    """Full-resolution, direction-aware boundary correction.

    For every region a soft morphological band is formed from the current
    probabilities.  The expert concatenates that band with a projection of the
    full-resolution feature, the probabilities, the boundary route weight and the
    image gradient, then predicts separate outward and inward actions.  Their
    signed sum is scaled by ``b_b`` and restricted to the band and the route
    weight, which concentrates the residual near the current surface.
    """

    def __init__(
        self,
        decoder_channels: int,
        num_outputs: int,
        has_regions: bool,
        feature_channels: int = 8,
        hidden_channels: int = 8,
        bound: float = 1.5,
        radius: int = 1,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.has_regions = has_regions
        self.bound = float(bound)
        self.radius = int(radius)
        self.use_checkpoint = bool(use_checkpoint)
        self.feature_projection = ConvNormAct(decoder_channels, feature_channels, 1, 0)

        band_channels = num_outputs if has_regions else 1
        in_channels = feature_channels + num_outputs + band_channels + 2
        self.stem = ContextStem(in_channels, hidden_channels)
        corrected_channels = num_outputs if has_regions else num_outputs - 1
        self.operator = ResidualOperator(
            hidden_channels, hidden_channels, 2 * corrected_channels, 'zero'
        )

    def project_feature(self, full_feature: torch.Tensor) -> torch.Tensor:
        """Project the full-resolution decoder feature to ``feature_channels``."""
        return self.feature_projection(full_feature)

    @staticmethod
    def prepare_image_gradient(image: torch.Tensor) -> torch.Tensor:
        """Pre-compute the image gradient, which is static across correction steps."""
        return image_gradient_cue(image)

    def forward(
        self,
        full_feature: torch.Tensor,
        current_logits: torch.Tensor,
        boundary_route: torch.Tensor,
        image: torch.Tensor,
        projected_feature: torch.Tensor | None = None,
        image_grad: torch.Tensor | None = None,
    ):
        """Return the boundary residual and its auxiliary tensors.

        Args:
            full_feature: Final decoder feature at the output resolution.
            current_logits: Region logits of the current step.
            boundary_route: Boundary action weight ``w_B`` from the diagnosis.
            image: Input patch, used when ``image_grad`` is not cached.
            projected_feature: Cached projection of ``full_feature``.
            image_grad: Cached image gradient magnitude.

        Returns:
            ``(residual, aux)`` where ``aux`` holds the band, the resampled route,
            the gating term and the two raw action maps.
        """
        spatial = full_feature.shape[2:]
        prob = (
            torch.sigmoid(current_logits)
            if self.has_regions
            else torch.softmax(current_logits, dim=1)
        )
        if prob.shape[2:] != spatial:
            prob = F.interpolate(prob, size=spatial, mode='trilinear', align_corners=False)

        route = F.interpolate(boundary_route, size=spatial, mode='trilinear', align_corners=False)
        if self.has_regions:
            band = soft_morphological_band(prob, self.radius)
        else:
            foreground = 1.0 - prob[:, :1]
            band = soft_morphological_band(foreground, self.radius)
        band_gate = band * route

        if image_grad is None:
            img = image
            if img.shape[2:] != spatial:
                img = F.interpolate(img, size=spatial, mode='trilinear', align_corners=False)
            image_grad = self.prepare_image_gradient(img)
        elif image_grad.shape[2:] != spatial:
            image_grad = F.interpolate(
                image_grad, size=spatial, mode='trilinear', align_corners=False
            )

        feature = (
            projected_feature
            if projected_feature is not None
            else self.project_feature(full_feature)
        )

        def dense_boundary_operator(feature_proj, prob_full, band_full, route_full, grad):
            hidden = self.stem(torch.cat([feature_proj, prob_full, band_full, route_full, grad], dim=1))
            return self.operator(hidden)

        if self.training and self.use_checkpoint:
            raw = checkpoint(
                dense_boundary_operator,
                feature,
                prob,
                band,
                route,
                image_grad,
                use_reentrant=False,
            )
        else:
            raw = dense_boundary_operator(feature, prob, band, route, image_grad)

        outward, inward = raw.chunk(2, dim=1)
        signed = self.bound * (torch.sigmoid(outward) - torch.sigmoid(inward))
        residual = band_gate * signed

        return residual, {
            'soft_band': band,
            'boundary_route_full': route,
            'band_gate': band_gate,
            'outward_raw': outward,
            'inward_raw': inward,
        }
