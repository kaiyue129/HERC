"""HERC attached to a frozen SSDMoE-UNet segmenter.

``HERCUNet`` is a strict extension of ``SSDMoEUNet``: every baseline module keeps
its name and shape, so a fold-specific SSDMoE-UNet checkpoint loads directly and
only the parameters under the ``herc.`` prefix are new.  This mirrors Section 3.1
of the manuscript, where the corrector reads the penultimate decoder feature
(64 channels at 64^3), the full-resolution decoder feature (32 channels at 128^3)
and the current region logits, while the backbone stays frozen.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Type, Union

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from nnunetv2.nets.herc.cell import HERCCorrector
from nnunetv2.nets.herc.config import HERCConfig
from nnunetv2.nets.herc.error_augmentation import inject_typed_error
from nnunetv2.nets.SSDMoEUNet import SSDMoEUNet


def highest_resolution(output):
    """Return the full-resolution tensor of a (possibly deep-supervised) output."""
    return output[0] if isinstance(output, (list, tuple)) else output


def replace_highest_resolution(output, new_high):
    """Replace the full-resolution tensor while keeping deep-supervision structure."""
    if isinstance(output, tuple):
        return (new_high, *output[1:])
    if isinstance(output, list):
        return [new_high, *output[1:]]
    return new_high


class HERCUNet(SSDMoEUNet):
    """Frozen SSDMoE-UNet plus the shared HERC correction cell.

    The class keeps all baseline state-dict keys unchanged (``encoder.*``,
    ``gsaf_modules.*``, ``ssdmoe_bottleneck.*``, ``decoder.*``); new parameters
    appear only under ``herc.*``.
    """

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes,
        strides,
        n_conv_per_stage,
        num_classes: int,
        n_conv_per_stage_decoder,
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        block=None,
        bottleneck_channels=None,
        stem_channels=None,
        has_regions: bool = True,
        herc_config: Optional[HERCConfig] = None,
    ):
        kwargs = dict(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
            bottleneck_channels=bottleneck_channels,
            stem_channels=stem_channels,
        )
        if block is not None:
            kwargs['block'] = block
        super().__init__(**kwargs)

        if len(self.decoder.stage_output_channels) < 2:
            raise RuntimeError('HERC requires at least two decoder stages.')
        coarse_channels = self.decoder.stage_output_channels[-2]
        full_channels = self.decoder.stage_output_channels[-1]

        cfg = herc_config or HERCConfig(
            num_outputs=num_classes,
            has_regions=has_regions,
        )
        cfg.num_outputs = num_classes
        cfg.has_regions = has_regions
        cfg.validate()
        self.herc_config = cfg
        self.herc = HERCCorrector(coarse_channels, full_channels, cfg)
        self.herc_enabled = True

    @property
    def has_regions(self) -> bool:
        """Whether the region-based BraTS contract is active."""
        return self.herc_config.has_regions

    def set_herc_enabled(self, enabled: bool) -> None:
        """Toggle the correction stage without touching the frozen backbone."""
        self.herc_enabled = bool(enabled)

    def set_baseline_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze every baseline parameter, leaving HERC untouched."""
        for name, parameter in self.named_parameters():
            if not name.startswith('herc.'):
                parameter.requires_grad = trainable

    def set_herc_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze the correction parameters."""
        for parameter in self.herc.parameters():
            parameter.requires_grad = trainable

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
        synthetic_error_target=None,
        synthetic_error_type: Optional[str] = None,
        correction_steps: Optional[int] = None,
    ):
        """Segment, then correct the residual errors of the segmentation.

        Args:
            x: Input patch ``[B, C, D, H, W]``.
            return_aux: Return the diagnostic dictionary instead of the outputs.
            synthetic_error_target: Reference used by the optional typed-error warm-up to inject
                a typed error into the initial logits.
            synthetic_error_type: Error type of the warm-up injection.
            correction_steps: Overrides the configured number of steps.

        Returns:
            The segmentation output, or a dictionary with the corrected, initial
            and baseline outputs plus the correction states when ``return_aux``.
        """
        base_output, features = super().forward(x, return_features=True)
        base_logits = highest_resolution(base_output)

        if not self.herc_enabled:
            if return_aux:
                return {
                    'seg_outputs': base_output,
                    'baseline_outputs': base_output,
                    'initial_outputs': base_output,
                    'herc': None,
                    'synthetic_error_type': None,
                }
            return base_output

        initial_logits = base_logits
        used_error_type = None
        if synthetic_error_target is not None:
            if self.has_regions:
                initial_logits, used_error_type = inject_typed_error(
                    base_logits,
                    synthetic_error_target,
                    has_regions=self.has_regions,
                    mode=synthetic_error_type,
                    wt=self.herc_config.wt_index,
                    tc=self.herc_config.tc_index,
                    et=self.herc_config.et_index,
                )
            else:
                initial_logits, used_error_type = inject_typed_error(
                    base_logits,
                    synthetic_error_target,
                    has_regions=False,
                    mode=synthetic_error_type,
                )

        corrected = self.herc(
            image=x,
            coarse_feature=features['decoder_coarse'],
            full_feature=features['decoder_full'],
            initial_logits=initial_logits,
            steps=correction_steps,
        )
        corrected_outputs = replace_highest_resolution(base_output, corrected['final_logits'])
        initial_outputs = replace_highest_resolution(base_output, initial_logits)

        if not return_aux:
            return corrected_outputs

        return {
            'seg_outputs': corrected_outputs,
            'baseline_outputs': base_output,
            'initial_outputs': initial_outputs,
            'herc': corrected,
            'synthetic_error_type': used_error_type,
            'feature_shapes': {
                'decoder_coarse': tuple(features['decoder_coarse'].shape),
                'decoder_full': tuple(features['decoder_full'].shape),
            },
        }
