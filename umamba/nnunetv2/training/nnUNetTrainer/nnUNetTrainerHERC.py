"""nnU-Net trainers for the HERC correction stage.

Section 3.3 of the manuscript trains the corrector while the segmenter stays
frozen: four-channel 128^3 patches, batch size one, 250 patch draws per epoch, 500
epochs per fold-specific run (125,000 optimizer steps), SGD with Nesterov momentum
0.99, an initial learning rate of 1e-2, a weight decay of 3e-5 and a polynomial
decay with exponent 0.9.

:class:`nnUNetTrainerHERC` reproduces that configuration.  The warm-up and joint
variants in this module are additional entry points and are not part of the
reported configuration.

The trainers read their settings from ``HERC_*`` environment variables so that one
code path covers every reported configuration.
"""

from __future__ import annotations

import os

import torch
from torch import nn, autocast

from dynamic_network_architectures.building_blocks.helper import (
    convert_dim_to_conv_op,
    get_matching_instancenorm,
)

from nnunetv2.nets.herc.config import HERCConfig
from nnunetv2.nets.HERCUNet import HERCUNet
from nnunetv2.training.loss.herc_loss import HERCCompositeLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSSDMoEUNet import (
    CustomPolyLRScheduler,
    nnUNetTrainerSSDMoEUNet,
)
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


def infer_nested_region_indices(label_manager):
    """Return the ``(WT, TC, ET)`` head indices of a nested region label manager.

    The indices are inferred from the size of the region sets instead of being
    hard-coded, so the same trainer works for any WT >= TC >= ET label definition.
    """
    if not label_manager.has_regions:
        return 0, 1, 2
    regions = label_manager.foreground_regions
    if len(regions) != 3:
        raise ValueError(
            f'HERC expects exactly three nested foreground regions; got {regions}'
        )
    sets = []
    for region in regions:
        if isinstance(region, (tuple, list)):
            sets.append({int(x) for x in region})
        else:
            sets.append({int(region)})
    order = sorted(range(3), key=lambda i: len(sets[i]), reverse=True)
    wt, tc, et = order
    if not (sets[et].issubset(sets[tc]) and sets[tc].issubset(sets[wt])):
        raise ValueError(f'Regions are not a WT >= TC >= ET nested chain: {regions}')
    return wt, tc, et


class nnUNetTrainerHERC(nnUNetTrainerSSDMoEUNet):
    """Main phase: natural residual errors, frozen SSDMoE-UNet, two correction steps."""

    use_synthetic_error_warmup = False
    joint_level = 0
    default_phase_epochs = 200
    default_train_steps = 2

    def __init__(
        self,
        plans,
        configuration,
        fold,
        dataset_json,
        unpack_dataset=True,
        device=torch.device('cuda'),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = int(os.environ.get('HERC_EPOCHS', self.default_phase_epochs))
        self.initial_lr = float(os.environ.get('HERC_LR', '1e-2'))
        self.weight_decay = float(os.environ.get('HERC_WEIGHT_DECAY', '3e-5'))
        self._phase_name = 'natural'

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        """Build the frozen SSDMoE-UNet with the shared HERC correction cell."""
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads
        num_stages = len(configuration_manager.conv_kernel_sizes)
        dim = len(configuration_manager.conv_kernel_sizes[0])
        conv_op = convert_dim_to_conv_op(dim)
        if dim != 3:
            raise NotImplementedError('HERC is defined for the 3D SSDMoE pipeline.')

        features_per_stage = [
            min(
                configuration_manager.UNet_base_num_features * 2 ** i,
                configuration_manager.unet_max_num_features,
            )
            for i in range(num_stages)
        ]

        wt_index, tc_index, et_index = infer_nested_region_indices(label_manager)
        cfg = HERCConfig(
            num_outputs=num_classes,
            has_regions=label_manager.has_regions,
            wt_index=wt_index,
            tc_index=tc_index,
            et_index=et_index,
            train_steps=int(os.environ.get('HERC_TRAIN_STEPS', '2')),
            max_inference_steps=int(os.environ.get('HERC_MAX_INFERENCE_STEPS', '3')),
            adaptive_stopping=os.environ.get('HERC_ADAPTIVE_STOP', '1').lower() in ('1', 'true', 'yes'),
            stopping_threshold=float(os.environ.get('HERC_STOP_THRESHOLD', '0.025')),
            stopping_topk_fraction=float(os.environ.get('HERC_STOP_TOPK_FRACTION', '0.02')),
        )

        return HERCUNet(
            input_channels=num_input_channels,
            n_stages=num_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=configuration_manager.conv_kernel_sizes,
            strides=configuration_manager.pool_op_kernel_sizes,
            n_conv_per_stage=configuration_manager.n_conv_per_stage_encoder,
            num_classes=num_classes,
            n_conv_per_stage_decoder=configuration_manager.n_conv_per_stage_decoder,
            conv_bias=True,
            norm_op=get_matching_instancenorm(conv_op),
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            deep_supervision=enable_deep_supervision,
            has_regions=label_manager.has_regions,
            herc_config=cfg,
        )

    def _model_before_ddp(self):
        return self.network

    @staticmethod
    def _set_trainable(module, value: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = value

    def _configure_trainable_groups(self):
        """Freeze the backbone and build one parameter group per active stage."""
        model = self._model_before_ddp()
        for parameter in model.parameters():
            parameter.requires_grad = False

        groups = []
        # The correction parameters are trainable in every phase.
        self._set_trainable(model.herc, True)
        groups.append({
            'params': [p for p in model.herc.parameters() if p.requires_grad],
            'lr': self.initial_lr,
            'weight_decay': self.weight_decay,
            'name': 'herc',
        })

        if self.joint_level >= 1:
            self._set_trainable(model.decoder.seg_layers, True)
            groups.append({
                'params': [p for p in model.decoder.seg_layers.parameters() if p.requires_grad],
                'lr': self.initial_lr * float(os.environ.get('HERC_HEAD_LR_RATIO', '0.1')),
                'weight_decay': self.weight_decay,
                'name': 'seg_heads',
            })

        if self.joint_level >= 2:
            n_stages = int(os.environ.get('HERC_JOINT_DECODER_STAGES', '1'))
            n_stages = max(1, min(n_stages, len(model.decoder.stages)))
            modules = list(model.decoder.stages[-n_stages:]) + list(
                model.decoder.transpconvs[-n_stages:]
            )
            groups.append({
                'params': self._unique_parameters(modules),
                'lr': self.initial_lr * float(os.environ.get('HERC_DECODER_LR_RATIO', '0.05')),
                'weight_decay': self.weight_decay,
                'name': 'late_decoder',
            })

        if self.joint_level >= 3:
            # Optional: tune the highest-resolution GSAF blocks only.  The
            # semantic SSD-MoE bottleneck stays frozen.
            n_gsaf = min(2, len(model.gsaf_modules))
            groups.append({
                'params': self._unique_parameters(list(model.gsaf_modules[:n_gsaf])),
                'lr': self.initial_lr * float(os.environ.get('HERC_GSAF_LR_RATIO', '0.02')),
                'weight_decay': self.weight_decay,
                'name': 'highres_gsaf',
            })

        # SSD-MoE stays frozen unless the optional ablation explicitly opens it.
        if os.environ.get('HERC_TUNE_SSDMOE', '0').lower() in ('1', 'true', 'yes'):
            routing, normal = [], []
            for name, parameter in model.ssdmoe_bottleneck.named_parameters():
                parameter.requires_grad = True
                if '.moe.phi' in name or '.moe.temperature' in name:
                    routing.append(parameter)
                else:
                    normal.append(parameter)
            if normal:
                groups.append({
                    'params': normal,
                    'lr': self.initial_lr * float(os.environ.get('HERC_SSDMOE_LR_RATIO', '0.01')),
                    'weight_decay': self.weight_decay,
                    'name': 'ssdmoe_optional',
                })
            if routing:
                groups.append({
                    'params': routing,
                    'lr': self.initial_lr
                    * float(os.environ.get('HERC_SSDMOE_ROUTING_LR_RATIO', '0.1')),
                    'weight_decay': 0.0,
                    'name': 'ssdmoe_routing_optional',
                })

        return groups

    def _unique_parameters(self, modules):
        """Unfreeze ``modules`` and return their parameters without duplicates."""
        params, seen = [], set()
        for module in modules:
            self._set_trainable(module, True)
            for parameter in module.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    seen.add(id(parameter))
                    params.append(parameter)
        return params

    def configure_optimizers(self):
        """Build the grouped SGD optimiser and the polynomial schedule."""
        groups = [g for g in self._configure_trainable_groups() if len(g['params']) > 0]
        optimizer = torch.optim.SGD(groups, momentum=0.99, nesterov=True)
        scheduler = CustomPolyLRScheduler(optimizer, self.num_epochs, exponent=0.9)
        self.print_to_log_file('--> HERC optimizer groups:')
        for group in groups:
            self.print_to_log_file(
                f"    {group['name']}: {len(group['params'])} tensors, "
                f"lr={group['lr']}, wd={group['weight_decay']}"
            )
        return optimizer, scheduler

    def _build_loss(self):
        """Wrap the nnU-Net loss into :class:`HERCCompositeLoss`."""
        base_loss = super()._build_loss()
        return HERCCompositeLoss(
            base_loss,
            has_regions=self.label_manager.has_regions,
            lambda_severity=float(os.environ.get('HERC_LAMBDA_SEVERITY', '0.20')),
            lambda_route=float(os.environ.get('HERC_LAMBDA_ROUTE', '0.10')),
            lambda_boundary=float(os.environ.get('HERC_LAMBDA_BOUNDARY', '0.10')),
            lambda_monotonic=float(os.environ.get('HERC_LAMBDA_MONOTONIC', '0.20')),
            lambda_safety=float(os.environ.get('HERC_LAMBDA_SAFETY', '0.10')),
            lambda_hierarchy=float(os.environ.get('HERC_LAMBDA_HIERARCHY', '0.05')),
            lambda_anchor=float(os.environ.get(
                'HERC_LAMBDA_ANCHOR', '0.05' if self.joint_level > 0 else '0.0'
            )),
            lambda_coarse=float(os.environ.get(
                'HERC_LAMBDA_COARSE', '0.10' if self.joint_level > 0 else '0.0'
            )),
        )

    def initialize(self):
        super().initialize()
        real_model = self._get_real_model()
        real_model.herc_config.train_steps = int(
            os.environ.get('HERC_TRAIN_STEPS', self.default_train_steps)
        )
        self.print_to_log_file(
            f'HERC phase={self._phase_name}, epochs={self.num_epochs}, '
            f'train_steps={real_model.herc_config.train_steps}, '
            f'regions={real_model.has_regions}'
        )

    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
            target_hi = target[0]
        else:
            target = target.to(self.device, non_blocking=True)
            target_hi = target

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(
                data,
                return_aux=True,
                synthetic_error_target=(target_hi if self.use_synthetic_error_warmup else None),
                correction_steps=(1 if self.use_synthetic_error_warmup else None),
            )
            loss = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.network.parameters() if p.requires_grad],
                max_norm=1.0,
                norm_type=2,
            )
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.network.parameters() if p.requires_grad],
                max_norm=1.0,
                norm_type=2,
            )
            self.optimizer.step()

        result = {'loss': loss.detach().cpu().numpy()}
        if hasattr(self.loss, 'last_components'):
            for key, value in self.loss.last_components.items():
                result[f'herc_{key}'] = float(value.detach().cpu())
        return result


class nnUNetTrainerHERCWarmup(nnUNetTrainerHERC):
    """Optional entry point: train the corrective operators on typed errors.

    Not used by the reported configuration.
    """

    use_synthetic_error_warmup = True
    joint_level = 0
    default_phase_epochs = 50
    default_train_steps = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._phase_name = 'typed_error_warmup'


class nnUNetTrainerHERCJoint(nnUNetTrainerHERC):
    """Optional entry point: conservative joint fine-tuning of the correction stage.

    Not used by the reported configuration.
    """

    use_synthetic_error_warmup = False
    joint_level = 2
    default_phase_epochs = 100
    default_train_steps = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.joint_level = int(os.environ.get('HERC_JOINT_LEVEL', '2'))
        self._phase_name = f'joint_level_{self.joint_level}'


__all__ = [
    'nnUNetTrainerHERC',
    'nnUNetTrainerHERCWarmup',
    'nnUNetTrainerHERCJoint',
]
