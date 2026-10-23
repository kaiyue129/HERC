"""
nnUNetTrainerSSDMoEUNet

Target model: nnunetv2.nets.SSDMoEUNet

Design philosophy (aligned with Hetero-UNet / nnU-Net style):
  - Inherit nnU-Net trainer as much as possible.
  - Override ONLY:
      1) build_network_architecture -> instantiate your custom network (SSDMoEUNet)
      2) monitoring hooks + logging for gating/routing modules:
          - GSAF (skip gating): router probs/entropy + forced-residual effectiveness
          - SoftMoE (in bottleneck Mamba2 blocks): dispatch/combine entropy + expert usage stats
          - Multi-kernel Stage0–1 encoder: fuse_out magnitude + fuse conv weight norm growth (zero-init sanity)
      3) [NEW] configure_optimizers -> inject a configurable LR multiplier & 0 weight decay
                                       for SoftMoE routing params.

Notes:
  - nnUNetLogger.log asserts keys must exist; we pre-register custom keys in initialize().
  - forward hooks + torch.compile can be brittle; we disable compile by default in this trainer.
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.optim.lr_scheduler import _LRScheduler

# Base trainer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op

# Your model (release naming aligned with the manuscript)
from nnunetv2.nets.SSDMoEUNet import SSDMoEUNet


class CustomPolyLRScheduler(_LRScheduler):
    """
    Custom Poly LR Scheduler that supports multiple parameter groups.
    nnU-Net's default PolyLRScheduler strictly overrides ALL param groups with a single initial_lr.
    This custom version memorizes the initial_lr per parameter group (crucial for SoftMoE routing LR sweeps).
    """

    def __init__(self, optimizer, max_steps: int, exponent: float = 0.9):
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        # Memorize the initial learning rate for each param group
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        # Calculate the decay factor
        factor = (1 - current_step / self.max_steps) ** self.exponent

        # Apply the decay factor proportionally to each group's unique initial LR
        for initial_lr, param_group in zip(self.initial_lrs, self.optimizer.param_groups):
            param_group['lr'] = initial_lr * factor


class nnUNetTrainerSSDMoEUNet(nnUNetTrainer):
    """
    Trainer for nnunetv2.nets.SSDMoEUNet.

    Monitoring:
      - GSAF router stats (entropy / mean weights / dominance)
      - GSAF forced-residual effectiveness (||Δ||/||x||)
      - SoftMoE routing stats:
          * dispatch entropy (softmax over tokens per slot)
          * combine entropy (softmax over slots per token)
          * expert usage entropy (how evenly experts are used)
          * max expert usage
      - Multi-kernel Stage0–1 fuse_out magnitude + fuse weight norm (sanity: zero-init fuse should grow)
    """

    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # keep nnU-Net defaults unless you want to change them
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5

        # Stats buffers (filled by forward hooks)
        self._monitor_mode: str = "train"  # "train" | "val"
        self.train_stats: Dict[str, List[float]] = defaultdict(list)
        self.val_stats: Dict[str, List[float]] = defaultdict(list)

        self._hooks_registered: bool = False

        # safer default: disable compile for this trainer because of hooks
        self._force_disable_compile: bool = True

        # SoftMoE routing controls
        self.routing_lr_multiplier = float(os.environ.get("SSDMOE_ROUTING_LR_MULTIPLIER", "10.0"))
        if self.routing_lr_multiplier <= 0:
            raise ValueError("SSDMOE_ROUTING_LR_MULTIPLIER must be positive.")

        # SoftMoE monitoring controls
        self._moe_max_tokens_for_stats: int = 1024

        # optional wandb
        try:
            import wandb  # noqa: F401
            self._wandb_available = True
        except Exception:
            self._wandb_available = False

    # ------------------------- compile control -------------------------
    def _do_i_compile(self):
        if self._force_disable_compile:
            return False
        return super()._do_i_compile()

    # ------------------------- init -------------------------
    def initialize(self):
        if not self.was_initialized:
            super().initialize()
            self._register_custom_logger_keys()
            self._register_monitoring_hooks()

    # ------------------------- OPTIMIZER OVERRIDE -------------------------
    def configure_optimizers(self):
        """
        Intercept the optimizer creation to isolate SoftMoE routing parameters.
        Provide them with 10x learning rate and 0 weight decay to break symmetry.
        """
        initial_lr = self.initial_lr
        weight_decay = self.weight_decay

        routing_params = []
        base_params = []

        # Isolate routing parameters (phi and temperature)
        for name, param in self.network.named_parameters():
            if not param.requires_grad:
                continue

            if "moe.phi" in name or "moe.temperature" in name:
                routing_params.append(param)
            else:
                base_params.append(param)

        # Group 0: Base network -> normal LR, normal WD
        # Group 1: Routing params -> configurable LR multiplier, 0 WD
        routing_lr = initial_lr * self.routing_lr_multiplier
        optimizer_groups = [
            {'params': base_params, 'lr': initial_lr, 'weight_decay': weight_decay},
            {'params': routing_params, 'lr': routing_lr, 'weight_decay': 0.0}
        ]

        # Standard nnU-Net optimizer settings (SGD with Nesterov momentum)
        optimizer = torch.optim.SGD(optimizer_groups, momentum=0.99, nesterov=True)

        # MUST use the CustomPolyLRScheduler to preserve the routing LR multiplier
        lr_scheduler = CustomPolyLRScheduler(optimizer, self.num_epochs, exponent=0.9)

        self.print_to_log_file(f"--> Optimizer configured with isolated routing parameters.")
        self.print_to_log_file(f"    Base params: {len(base_params)} tensors, lr={initial_lr}")
        self.print_to_log_file(
            f"    Routing params: {len(routing_params)} tensors, "
            f"lr={routing_lr} (multiplier={self.routing_lr_multiplier})"
        )

        return optimizer, lr_scheduler

    # ------------------------- utils -------------------------
    def _get_real_model(self) -> nn.Module:
        m = self.network.module if self.is_ddp else self.network
        if isinstance(m, OptimizedModule):
            m = m._orig_mod
        return m

    def _append_stat(self, key: str, value: float):
        if self._monitor_mode == "train":
            self.train_stats[key].append(float(value))
        else:
            self.val_stats[key].append(float(value))

    def _ddp_mean(self, value: float) -> float:
        if not (dist.is_available() and dist.is_initialized()):
            return float(value)
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, float(value))
        return float(np.mean(gathered))

    def _summarize_stats(self, stats: Dict[str, List[float]]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in stats.items():
            if len(v) == 0:
                continue
            out[k] = self._ddp_mean(float(np.mean(v)))
        return out

    # ------------------------- register logger keys -------------------------
    def _ensure_logger_key(self, key: str):
        if not hasattr(self, "logger") or self.logger is None:
            return
        if not hasattr(self.logger, "my_fantastic_logging"):
            return
        if key not in self.logger.my_fantastic_logging or not isinstance(self.logger.my_fantastic_logging[key], list):
            self.logger.my_fantastic_logging[key] = []

    def _register_custom_logger_keys(self):
        keys = [
            "train_gsaf_entropy_mean", "train_gsaf_resratio_l2_mean",
            "train_moe_dispatch_entropy_mean", "train_moe_combine_entropy_mean",
            "train_moe_expert_entropy_mean", "train_moe_expert_max_mean", "train_moe_phi_norm_mean",
            "train_mk_stage0_fuse_abs_mean", "train_mk_stage1_fuse_abs_mean",
            "train_mk_stage0_fuse_w_norm_mean", "train_mk_stage1_fuse_w_norm_mean",
            "val_gsaf_entropy_mean", "val_gsaf_resratio_l2_mean",
            "val_moe_dispatch_entropy_mean", "val_moe_combine_entropy_mean",
            "val_moe_expert_entropy_mean", "val_moe_expert_max_mean",
            "val_mk_stage0_fuse_abs_mean", "val_mk_stage1_fuse_abs_mean",
        ]
        for k in keys:
            self._ensure_logger_key(k)

    # ------------------------- hook factories -------------------------
    def _hook_router_stats(self, prefix: str, num_choices: int):
        def hook(module, inputs, output):
            with torch.no_grad():
                logits = output
                if not isinstance(logits, torch.Tensor):
                    return
                probs = F.softmax(logits, dim=1)
                for i in range(num_choices):
                    self._append_stat(f"{prefix}/w_mean_{i}", probs[:, i].mean().item())

                ent = -(probs * (probs + 1e-8).log()).sum(dim=1).mean().item()
                self._append_stat(f"{prefix}/entropy", ent)

                maxp = probs.max(dim=1).values.mean().item()
                self._append_stat(f"{prefix}/max_prob", maxp)

                arg = probs.argmax(dim=1)
                for i in range(num_choices):
                    dom = (arg == i).float().mean().item()
                    self._append_stat(f"{prefix}/dominance_{i}", dom)

        return hook

    def _hook_residual_effect(self, prefix: str, ref_input_index: int = 0):
        def hook(module, inputs, output):
            x = inputs[ref_input_index]
            y = output
            with torch.no_grad():
                if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.shape == y.shape):
                    return
                delta = y - x
                ratio_l2 = (torch.norm(delta) / (torch.norm(x) + 1e-6)).item()
                self._append_stat(f"{prefix}/res_ratio_l2", ratio_l2)

        return hook

    def _hook_abs_mean(self, prefix: str):
        def hook(module, inputs, output):
            with torch.no_grad():
                t = output
                if isinstance(t, (tuple, list)):
                    t = t[0]
                if not isinstance(t, torch.Tensor):
                    return
                self._append_stat(f"{prefix}/abs_mean", t.abs().mean().item())

        return hook

    def _hook_softmoe_stats(self, layer_idx: int):
        prefix = f"SoftMoE/Layer_{layer_idx}"

        def hook(module, inputs, output):
            with torch.no_grad():
                if not hasattr(module, "phi"):
                    return
                if len(inputs) == 0 or not isinstance(inputs[0], torch.Tensor):
                    return
                x = inputs[0]
                if x.ndim != 3:
                    return
                B, L, D = x.shape

                if L > self._moe_max_tokens_for_stats:
                    idx = torch.randperm(L, device=x.device)[: self._moe_max_tokens_for_stats]
                    x = x[:, idx, :]
                    L = x.shape[1]

                # If temperature scaling is present in the module, apply it to stats
                phi = module.phi
                temp = getattr(module, "temperature", 1.0)

                # Check if x is normalized inside the module, approximate here
                x_norm = F.normalize(x, p=2, dim=-1)
                phi_norm = F.normalize(phi, p=2, dim=0)

                logits = torch.einsum("bld,dnp->blnp", x_norm, phi_norm)
                logits = logits * temp

                dispatch = torch.softmax(logits, dim=1)
                combine = torch.softmax(logits.reshape(B, L, -1), dim=-1).reshape_as(logits)

                ent_dispatch = -(dispatch * (dispatch + 1e-8).log()).sum(dim=1)
                self._append_stat(f"{prefix}/dispatch_entropy", ent_dispatch.mean().item())

                ent_combine = -(combine.reshape(B, L, -1) * (combine.reshape(B, L, -1) + 1e-8).log()).sum(dim=-1)
                self._append_stat(f"{prefix}/combine_entropy", ent_combine.mean().item())

                usage = combine.sum(dim=-1)
                usage_mean = usage.mean(dim=(0, 1))
                usage_prob = usage_mean / (usage_mean.sum() + 1e-8)

                ent_expert = -(usage_prob * (usage_prob + 1e-8).log()).sum().item()
                self._append_stat(f"{prefix}/expert_entropy", ent_expert)

                max_expert = usage_prob.max().item()
                self._append_stat(f"{prefix}/expert_max", max_expert)

                self._append_stat(f"{prefix}/phi_norm", float(torch.norm(phi).detach().cpu().item()))

        return hook

    # ------------------------- register hooks -------------------------
    def _register_monitoring_hooks(self):
        if self._hooks_registered:
            return

        real_model = self._get_real_model()
        self.print_to_log_file("--> Registering monitoring hooks (GSAF + SoftMoE + Multi-kernel Stage0–1)...")

        if hasattr(real_model, "gsaf_modules"):
            for i, layer in enumerate(real_model.gsaf_modules):
                stage_name = f"GSAF/Stage_{i}"
                if hasattr(layer, "router"):
                    layer.router.register_forward_hook(self._hook_router_stats(stage_name + "/router", num_choices=2))
                layer.register_forward_hook(self._hook_residual_effect(stage_name, ref_input_index=0))

        bottleneck = getattr(real_model, "ssdmoe_bottleneck", None)
        if bottleneck is None:
            bottleneck = getattr(real_model, "mamba_mlp", None)
        if bottleneck is not None and hasattr(bottleneck, "layers"):
            for li, block in enumerate(bottleneck.layers):
                moe = getattr(block, "moe", None)
                if moe is None:
                    continue
                moe.register_forward_hook(self._hook_softmoe_stats(layer_idx=li))

        if hasattr(real_model, "encoder"):
            enc = real_model.encoder
            for sid in (0, 1):
                stage = getattr(enc, f"stage{sid}", None)
                if stage is None or not hasattr(stage, "blocks"):
                    continue
                for bi, blk in enumerate(stage.blocks):
                    if hasattr(blk, "fuse"):
                        blk.fuse.register_forward_hook(self._hook_abs_mean(f"MK/Stage{sid}/Block_{bi}/fuse_out"))

        self._hooks_registered = True
        self.print_to_log_file("--> Monitoring hooks registered.")

    # ------------------------- epoch lifecycle -------------------------
    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._monitor_mode = "train"
        self.train_stats.clear()

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._monitor_mode = "val"
        self.val_stats.clear()

    def _collect_weight_norms(self) -> Dict[str, float]:
        real_model = self._get_real_model()
        norms: Dict[str, float] = {}

        def _safe_norm(t: Optional[torch.Tensor]) -> float:
            if t is None:
                return 0.0
            return float(torch.norm(t.detach()).cpu().item())

        if hasattr(real_model, "gsaf_modules"):
            for i, layer in enumerate(real_model.gsaf_modules):
                try:
                    norms[f"GSAF/Stage_{i}/router_w_norm"] = _safe_norm(layer.router.weight)
                except Exception:
                    pass
                try:
                    norms[f"GSAF/Stage_{i}/delta_proj_w_norm"] = _safe_norm(layer.delta_proj.weight)
                except Exception:
                    pass

        if hasattr(real_model, "encoder"):
            enc = real_model.encoder
            for sid in (0, 1):
                stage = getattr(enc, f"stage{sid}", None)
                if stage is None or not hasattr(stage, "blocks"):
                    continue
                fuse_norms: List[float] = []
                for blk in stage.blocks:
                    try:
                        fuse_norms.append(_safe_norm(blk.fuse[0].weight))
                    except Exception:
                        continue
                if len(fuse_norms) > 0:
                    norms[f"MK/Stage{sid}/fuse_w_norm_mean"] = float(np.mean(fuse_norms))

        bottleneck = getattr(real_model, "ssdmoe_bottleneck", None)
        if bottleneck is None:
            bottleneck = getattr(real_model, "mamba_mlp", None)
        if bottleneck is not None and hasattr(bottleneck, "layers"):
            phi_norms: List[float] = []
            for block in bottleneck.layers:
                moe = getattr(block, "moe", None)
                if moe is None or not hasattr(moe, "phi"):
                    continue
                phi_norms.append(_safe_norm(moe.phi))
            if len(phi_norms) > 0:
                norms["SoftMoE/phi_norm_mean"] = float(np.mean(phi_norms))

        if dist.is_available() and dist.is_initialized():
            for k, v in list(norms.items()):
                norms[k] = self._ddp_mean(v)

        return norms

    def _log_compact_epoch_metrics(self, summary: Dict[str, float], norms: Optional[Dict[str, float]], prefix: str):
        def _mean_or_nan(vals: List[float]) -> float:
            return float(np.mean(vals)) if len(vals) > 0 else float("nan")

        gsaf_ent = [v for k, v in summary.items() if k.endswith("/router/entropy") and k.startswith("GSAF/Stage_")]
        gsaf_rr = [v for k, v in summary.items() if k.endswith("/res_ratio_l2") and k.startswith("GSAF/Stage_")]
        self.logger.log(f"{prefix}_gsaf_entropy_mean", _mean_or_nan(gsaf_ent), self.current_epoch)
        self.logger.log(f"{prefix}_gsaf_resratio_l2_mean", _mean_or_nan(gsaf_rr), self.current_epoch)

        moe_de = [v for k, v in summary.items() if k.endswith("/dispatch_entropy") and k.startswith("SoftMoE/Layer_")]
        moe_ce = [v for k, v in summary.items() if k.endswith("/combine_entropy") and k.startswith("SoftMoE/Layer_")]
        moe_ee = [v for k, v in summary.items() if k.endswith("/expert_entropy") and k.startswith("SoftMoE/Layer_")]
        moe_em = [v for k, v in summary.items() if k.endswith("/expert_max") and k.startswith("SoftMoE/Layer_")]

        self.logger.log(f"{prefix}_moe_dispatch_entropy_mean", _mean_or_nan(moe_de), self.current_epoch)
        self.logger.log(f"{prefix}_moe_combine_entropy_mean", _mean_or_nan(moe_ce), self.current_epoch)
        self.logger.log(f"{prefix}_moe_expert_entropy_mean", _mean_or_nan(moe_ee), self.current_epoch)
        self.logger.log(f"{prefix}_moe_expert_max_mean", _mean_or_nan(moe_em), self.current_epoch)

        if prefix == "train" and norms is not None:
            self.logger.log("train_moe_phi_norm_mean", float(norms.get("SoftMoE/phi_norm_mean", float("nan"))),
                            self.current_epoch)

        mk0 = [v for k, v in summary.items() if k.startswith("MK/Stage0/") and k.endswith("/fuse_out/abs_mean")]
        mk1 = [v for k, v in summary.items() if k.startswith("MK/Stage1/") and k.endswith("/fuse_out/abs_mean")]
        self.logger.log(f"{prefix}_mk_stage0_fuse_abs_mean", _mean_or_nan(mk0), self.current_epoch)
        self.logger.log(f"{prefix}_mk_stage1_fuse_abs_mean", _mean_or_nan(mk1), self.current_epoch)

        if prefix == "train" and norms is not None:
            self.logger.log("train_mk_stage0_fuse_w_norm_mean",
                            float(norms.get("MK/Stage0/fuse_w_norm_mean", float("nan"))),
                            self.current_epoch)
            self.logger.log("train_mk_stage1_fuse_w_norm_mean",
                            float(norms.get("MK/Stage1/fuse_w_norm_mean", float("nan"))),
                            self.current_epoch)

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)

        train_summary = self._summarize_stats(self.train_stats)
        norms = self._collect_weight_norms()

        self._log_compact_epoch_metrics(train_summary, norms=norms, prefix="train")

        if self.local_rank != 0:
            return

        if len(train_summary) > 0:
            self.print_to_log_file("\n" + "=" * 110)
            self.print_to_log_file(f"[Train Monitor - Epoch {self.current_epoch}] (GSAF + SoftMoE + MK Stage0–1)")
            self.print_to_log_file("-" * 110)

            gsaf_stages = sorted({k.split("/")[1] for k in train_summary.keys() if k.startswith("GSAF/Stage_")})
            if len(gsaf_stages) > 0:
                self.print_to_log_file("GSAF (Skip Gating: DCA vs PFAM):")
                self.print_to_log_file(f"{'Stage':<12} | {'w0':>7} {'w1':>7} | {'H':>7} {'maxP':>7} | {'Δ/x(L2)%':>10}")
                self.print_to_log_file("-" * 110)
                for st in gsaf_stages:
                    base = f"GSAF/{st}"
                    w0 = train_summary.get(base + "/router/w_mean_0", 0.0)
                    w1 = train_summary.get(base + "/router/w_mean_1", 0.0)
                    H = train_summary.get(base + "/router/entropy", 0.0)
                    mp = train_summary.get(base + "/router/max_prob", 0.0)
                    rr = train_summary.get(base + "/res_ratio_l2", 0.0) * 100.0
                    self.print_to_log_file(f"{st:<12} | {w0:7.4f} {w1:7.4f} | {H:7.4f} {mp:7.4f} | {rr:10.4f}")
                self.print_to_log_file("-" * 110)

            moe_layers = sorted({k.split("/")[1] for k in train_summary.keys() if k.startswith("SoftMoE/Layer_")})
            if len(moe_layers) > 0:
                self.print_to_log_file("SoftMoE (Bottleneck routing):")
                self.print_to_log_file(f"{'Layer':<12} | {'H_disp':>8} {'H_comb':>8} | {'H_exp':>8} {'max_exp':>8} | {'phi||':>10}")
                self.print_to_log_file("-" * 110)
                for lyr in moe_layers:
                    base = f"SoftMoE/{lyr}"
                    hd = train_summary.get(base + "/dispatch_entropy", float("nan"))
                    hc = train_summary.get(base + "/combine_entropy", float("nan"))
                    he = train_summary.get(base + "/expert_entropy", float("nan"))
                    mx = train_summary.get(base + "/expert_max", float("nan"))
                    pn = train_summary.get(base + "/phi_norm", float("nan"))
                    self.print_to_log_file(f"{lyr:<12} | {hd:8.4f} {hc:8.4f} | {he:8.4f} {mx:8.4f} | {pn:10.4f}")
                self.print_to_log_file("-" * 110)

            self.print_to_log_file("Multi-kernel Encoder (Stage0–1):")
            for sid in (0, 1):
                fuse_abs = [v for k, v in train_summary.items()
                            if k.startswith(f"MK/Stage{sid}/") and k.endswith("/fuse_out/abs_mean")]
                fuse_abs_mean = float(np.mean(fuse_abs)) if len(fuse_abs) > 0 else float("nan")
                fuse_w_norm_mean = float(norms.get(f"MK/Stage{sid}/fuse_w_norm_mean", float("nan")))
                self.print_to_log_file(f"  Stage{sid}: fuse_out |abs| mean={fuse_abs_mean:.6f} ; fuse_w_norm_mean={fuse_w_norm_mean:.6f}")

            if len(norms) > 0:
                self.print_to_log_file("Key weight norms (sanity):")
                for k in sorted(norms.keys()):
                    self.print_to_log_file(f"  {k}: {norms[k]:.6f}")

            self.print_to_log_file("=" * 110 + "\n")

        if self._wandb_available:
            try:
                import wandb
                log_dict = {}
                for k, v in train_summary.items():
                    log_dict["train/" + k] = v
                for k, v in norms.items():
                    log_dict["train/" + k] = v
                wandb.log(log_dict, step=self.current_epoch)
            except Exception:
                pass

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)

        val_summary = self._summarize_stats(self.val_stats)
        self._log_compact_epoch_metrics(val_summary, norms=None, prefix="val")

        if self.local_rank == 0 and self._wandb_available and len(val_summary) > 0:
            try:
                import wandb
                wandb.log({("val/" + k): v for k, v in val_summary.items()}, step=self.current_epoch)
            except Exception:
                pass

    # ------------------------- network construction -------------------------
    @staticmethod
    def build_network_architecture(
            plans_manager: PlansManager,
            dataset_json,
            configuration_manager: ConfigurationManager,
            num_input_channels,
            enable_deep_supervision: bool = True,
    ) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_classes = label_manager.num_segmentation_heads

        num_stages = len(configuration_manager.conv_kernel_sizes)
        dim = len(configuration_manager.conv_kernel_sizes[0])
        conv_op = convert_dim_to_conv_op(dim)

        features_per_stage = [
            min(configuration_manager.UNet_base_num_features * 2 ** i, configuration_manager.unet_max_num_features)
            for i in range(num_stages)
        ]

        model = SSDMoEUNet(
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
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=enable_deep_supervision,
        )
        return model

# Backward-compatible alias
nnUNetTrainer7 = nnUNetTrainerSSDMoEUNet

__all__ = ["nnUNetTrainerSSDMoEUNet", "CustomPolyLRScheduler"]
