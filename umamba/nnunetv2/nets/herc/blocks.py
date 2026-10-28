"""Low-level 3D building blocks shared by the HERC components.

All modules are volumetric by construction, so no topology suffix is used in the
class names.  Every block keeps the parameter count of the reported model: the
context stems are depthwise separable, and the residual operators zero-initialise
their output projection so that the first correction step is a no-op.
"""

from __future__ import annotations

import torch
from torch import nn


def _valid_groups(channels: int, preferred: int = 8) -> int:
    """Largest group count from ``(preferred, 4, 2, 1)`` that divides ``channels``."""
    for groups in (preferred, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    """3x3 (or 1x1) convolution followed by group normalisation and SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        groups: int = 1,
        activation: bool = True,
    ):
        layers = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(_valid_groups(out_channels), out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class ResidualStem(nn.Module):
    """Two stacked :class:`ConvNormAct` blocks with an identity or 1x1 skip."""

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_channels, hidden_channels),
            ConvNormAct(hidden_channels, hidden_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == hidden_channels
            else nn.Conv3d(in_channels, hidden_channels, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class ContextStem(nn.Module):
    """Low-compute context extractor: 1x1 projection plus depthwise 3x3x3 mixing.

    Section 3.2.2 of the manuscript describes this stem as "pointwise and
    depthwise 3 x 3 x 3 convolutions".
    """

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.in_proj = ConvNormAct(in_channels, hidden_channels, 1, 0)
        self.dw1 = ConvNormAct(hidden_channels, hidden_channels, 3, 1, groups=hidden_channels)
        self.pw = ConvNormAct(hidden_channels, hidden_channels, 1, 0)
        self.dw2 = ConvNormAct(hidden_channels, hidden_channels, 3, 1, groups=hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.dw1(x)
        x = self.pw(x)
        return self.dw2(x)


class ResidualOperator(nn.Module):
    """Depthwise-separable 3D residual operator used by the typed experts.

    Args:
        output_init: ``'zero'`` yields exactly zero output at initialisation,
            ``'small_negative'`` yields a small negative output, which keeps the
            signed experts inactive until supervision moves them.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        output_init: str = 'zero',
    ):
        super().__init__()
        self.in_proj = ConvNormAct(in_channels, hidden_channels, 1, 0)
        self.dw1 = ConvNormAct(hidden_channels, hidden_channels, 3, 1, groups=hidden_channels)
        self.pw = ConvNormAct(hidden_channels, hidden_channels, 1, 0)
        self.dw2 = ConvNormAct(hidden_channels, hidden_channels, 3, 1, groups=hidden_channels)
        self.out_proj = nn.Conv3d(hidden_channels, out_channels, 1, bias=True)

        if output_init == 'zero':
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        elif output_init == 'small_negative':
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-3)
            nn.init.constant_(self.out_proj.bias, -4.0)
        else:
            raise ValueError(f'Unknown output_init: {output_init}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.dw1(x)
        x = self.pw(x)
        x = self.dw2(x)
        return self.out_proj(x)
