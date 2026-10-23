"""
SSDMoE-UNet release version.

This file keeps the original implementation logic but aligns public naming
with the paper terminology for repository release and reproducibility.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Union, Type, List, Tuple
import math

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from nnunetv2.utilities.network_initialization import InitWeights_He
import mamba_ssm
from mamba_ssm import Mamba2
from torch.nn import MultiheadAttention as MHA

# ==============================================================
#  Mamba2 / causal-conv1d stride alignment helper
#  The fused causal_conv1d kernel used by Mamba2 requires certain stride alignment
#  (e.g., x.stride(0) and x.stride(2) multiples of 8). We enforce a compatible
#  layout and pad sequence length to a multiple of 8 when needed.
# ==============================================================

def _choose_mamba2_headdim(d_model: int, min_heads: int = 8) -> int:
    """
    Pick a headdim that:
      - divides d_model
      - yields at least `min_heads` heads (d_model / headdim >= min_heads) when possible
      - prefers typical values used in practice (64, 48, 40, 32, 24, 16, 8)
    This helps satisfy causal_conv1d stride assertions that can trigger when the number of heads is small.
    """
    if d_model <= 0:
        return 8
    max_headdim = max(1, d_model // max(1, min_heads))
    preferred = [64, 48, 40, 32, 24, 16, 8]
    for h in preferred:
        if h <= max_headdim and d_model % h == 0:
            return h
    # fallback: find any divisor <= max_headdim
    for h in range(max_headdim, 0, -1):
        if d_model % h == 0:
            return h
    return 8

def _mamba2_prepare(x: torch.Tensor, multiple: int = 8):
    """
    Prepare (B, L, C) tensor for Mamba2 fused kernels:
      1) Pad L to a multiple of `multiple` (zeros at the end; safe for causal ops when we slice back).
      2) Force a layout where stride(1)=1 and stride(2)=L_padded by doing:
           x = (x^T).contiguous().T
    Returns:
      x_prepared, orig_L
    """
    assert x.dim() == 3, f"Expected (B, L, C), got {tuple(x.shape)}"
    B, L, C = x.shape
    pad = (-L) % multiple
    if pad:
        # pad on sequence dimension (middle dim): (pad_left, pad_right) for last dim first
        x = F.pad(x, (0, 0, 0, pad))
    # make "channel-last-like" 1D layout for causal_conv1d: stride(1)=1, stride(2)=L_padded
    x = x.transpose(1, 2).contiguous().transpose(1, 2)
    return x, L



# ==============================================================
#  Multi-Kernel Parallel Encoder Blocks (Stage0-1)
#  Inspired by multi-kernel / multi-receptive-field designs such as
#  Inception-style blocks and SKNet-like adaptive receptive fields.
#  (We implement the deterministic multi-branch variant; routing/gating
#  can be added on top if desired.)
# ==============================================================
def _to_3tuple(v):
    """Convert int/tuple/list to a 3-tuple (D,H,W)."""
    if isinstance(v, (tuple, list)):
        assert len(v) == 3, f"Expected 3D tuple/list, got {v}"
        return tuple(int(x) for x in v)
    return (int(v), int(v), int(v))


def _get_padding(kernel_size, dilation=1):
    ks = _to_3tuple(kernel_size)
    dil = _to_3tuple(dilation)
    # same padding for odd kernels
    return tuple(((k - 1) // 2) * d for k, d in zip(ks, dil))


class ConvNormNonlin(nn.Module):
    """Conv -> Norm -> Nonlin (+ optional Dropout)."""
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        dilation=1,
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: Union[None, dict] = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: Union[None, dict] = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: Union[None, dict] = None,
    ):
        super().__init__()
        stride = _to_3tuple(stride)
        dilation = _to_3tuple(dilation)
        padding = _get_padding(kernel_size, dilation)

        self.conv = conv_op(
            in_channels, out_channels,
            kernel_size=_to_3tuple(kernel_size),
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=conv_bias,
        )

        self.norm = None
        if norm_op is not None:
            self.norm = norm_op(out_channels, **(norm_op_kwargs or {}))

        self.dropout = None
        if dropout_op is not None:
            self.dropout = dropout_op(**(dropout_op_kwargs or {}))

        if nonlin is None:
            self.nonlin = nn.LeakyReLU(inplace=True)
        else:
            self.nonlin = nonlin(**(nonlin_kwargs or {}))

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.nonlin(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class MultiKernelResidualBlock(nn.Module):
    """
    Multi-branch (parallel) multi-kernel block with forced residual:
        y = x_proj + Fusion([B1(x), B2(x), B3(x)])
    Branches (default):
      - Local:   kxkxk conv
      - Dilated: kxkxk conv with dilation=2
      - Aniso:   (1,k,k) then (k,1,1)
    """
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: Union[None, dict] = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: Union[None, dict] = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: Union[None, dict] = None,
        dilated_d: int = 2,
        zero_init_fusion: bool = True,
    ):
        super().__init__()
        stride = _to_3tuple(stride)
        ks = _to_3tuple(kernel_size)

        # channel split for branches (sum to out_channels)
        c1 = out_channels // 3
        c2 = out_channels // 3
        c3 = out_channels - c1 - c2

        # Branch 1: local conv
        self.branch_local = ConvNormNonlin(
            conv_op, in_channels, c1,
            kernel_size=ks,
            stride=stride,
            dilation=1,
            conv_bias=conv_bias,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs
        )

        # Branch 2: dilated conv (same kernel, larger receptive field)
        self.branch_dilated = ConvNormNonlin(
            conv_op, in_channels, c2,
            kernel_size=ks,
            stride=stride,
            dilation=(dilated_d, dilated_d, dilated_d),
            conv_bias=conv_bias,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs
        )

        # Branch 3: anisotropic factorized conv
        # Use stride in the first conv to match downsampling behavior
        self.branch_aniso_1 = ConvNormNonlin(
            conv_op, in_channels, c3,
            kernel_size=(1, ks[1], ks[2]),
            stride=stride,
            dilation=1,
            conv_bias=conv_bias,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs
        )
        self.branch_aniso_2 = ConvNormNonlin(
            conv_op, c3, c3,
            kernel_size=(ks[0], 1, 1),
            stride=1,
            dilation=1,
            conv_bias=conv_bias,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs
        )

        # Fusion projection (1x1) to mix branch features
        self.fuse = nn.Sequential(
            conv_op(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            (norm_op(out_channels, **(norm_op_kwargs or {})) if norm_op is not None else nn.Identity()),
        )
        if zero_init_fusion:
            # start close to identity through residual path
            nn.init.constant_(self.fuse[0].weight, 0.0)

        # Residual projection if needed (channel/stride mismatch)
        needs_proj = (in_channels != out_channels) or (stride != (1, 1, 1))
        self.proj = None
        if needs_proj:
            self.proj = conv_op(
                in_channels, out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False
            )

        # Final nonlinearity (kept separate from branches)
        if nonlin is None:
            self.out_nonlin = nn.LeakyReLU(inplace=True)
        else:
            self.out_nonlin = nonlin(**(nonlin_kwargs or {}))

    def forward(self, x):
        residual = x if self.proj is None else self.proj(x)

        b1 = self.branch_local(x)
        b2 = self.branch_dilated(x)
        b3 = self.branch_aniso_2(self.branch_aniso_1(x))

        fused = torch.cat([b1, b2, b3], dim=1)  # [B, out_channels, D, H, W]
        fused = self.fuse(fused)

        out = residual + fused
        return self.out_nonlin(out)


class StackedMultiKernelResidualBlocks(nn.Module):
    """Stack N multi-kernel residual blocks; first block can downsample via stride."""
    def __init__(
        self,
        n_blocks: int,
        conv_op: Type[_ConvNd],
        input_channels: int,
        output_channels: int,
        kernel_size,
        initial_stride,
        conv_bias: bool,
        norm_op,
        norm_op_kwargs,
        dropout_op,
        dropout_op_kwargs,
        nonlin,
        nonlin_kwargs,
    ):
        super().__init__()
        assert n_blocks >= 1, "n_blocks must be >= 1"
        blocks = []
        for i in range(n_blocks):
            blocks.append(
                MultiKernelResidualBlock(
                    conv_op=conv_op,
                    in_channels=input_channels if i == 0 else output_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    stride=initial_stride if i == 0 else 1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class ShallowMultiKernelEncoder(nn.Module):
    """
    A drop-in replacement for ResidualEncoder that:
      - Implements Stage0-1 with multi-kernel parallel residual blocks
      - Delegates Stage2+ to the standard ResidualEncoder (to minimize changes)
    It exposes the same attributes needed by UNetResDecoder.
    """
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: List[int],
        conv_op: Type[_ConvNd],
        kernel_sizes: List[Union[int, Tuple[int, int, int]]],
        strides: List[Union[int, Tuple[int, int, int]]],
        n_conv_per_stage: Union[int, List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: Union[None, dict] = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: Union[None, dict] = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: Union[None, dict] = None,
        block: Union[None, Type[nn.Module]] = None,  # kept for interface compatibility
        bottleneck_channels=None,
        return_skips: bool = True,
        disable_default_stem: bool = False,
        stem_channels=None,
        mk_stages: Tuple[int, int] = (0, 1),
    ):
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have length n_stages"
        assert n_stages >= 2, "ShallowMultiKernelEncoder requires at least 2 stages"

        self.return_skips = return_skips

        # Public attrs consumed by UNetResDecoder
        self.output_channels = list(features_per_stage)
        self.conv_op = conv_op
        self.kernel_sizes = list(kernel_sizes)
        self.strides = list(strides)
        self.conv_bias = conv_bias
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs

        # Stage0-1: multi-kernel blocks
        self.stage0 = StackedMultiKernelResidualBlocks(
            n_blocks=n_conv_per_stage[0],
            conv_op=conv_op,
            input_channels=input_channels,
            output_channels=features_per_stage[0],
            kernel_size=kernel_sizes[0],
            initial_stride=strides[0],
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        self.stage1 = StackedMultiKernelResidualBlocks(
            n_blocks=n_conv_per_stage[1],
            conv_op=conv_op,
            input_channels=features_per_stage[0],
            output_channels=features_per_stage[1],
            kernel_size=kernel_sizes[1],
            initial_stride=strides[1],
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

        # Stage2+: reuse ResidualEncoder to keep parity with baseline
        self.deep_encoder = None
        if n_stages > 2:
            self.deep_encoder = ResidualEncoder(
                features_per_stage[1],                # input channels to Stage2
                n_stages - 2,
                features_per_stage[2:],
                conv_op,
                kernel_sizes[2:],
                strides[2:],
                n_conv_per_stage[2:],
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                block if block is not None else BasicBlockD,
                bottleneck_channels,
                return_skips=True,
                disable_default_stem=True,  # we already have Stage0-1
                stem_channels=stem_channels
            )

    def forward(self, x):
        skips = []

        x = self.stage0(x)
        skips.append(x)

        x = self.stage1(x)
        skips.append(x)

        if self.deep_encoder is not None:
            deeper_skips = self.deep_encoder(x)
            skips.extend(deeper_skips)

        return skips if self.return_skips else skips[-1]

    def compute_conv_feature_map_size(self, input_size):
        """
        Rough VRAM proxy: sum over stage outputs of (C * prod(spatial)).
        This is only used for planning/estimation and doesn't affect forward correctness.
        """
        # input_size: (D,H,W)
        size = np.int64(0)
        shp = np.array(input_size, dtype=np.int64)
        for c, st in zip(self.output_channels, self.strides):
            st = np.array(_to_3tuple(st), dtype=np.int64)
            shp = (shp + st - 1) // st  # ceil divide
            size += np.int64(c) * np.int64(np.prod(shp))
        return size


# ==============================================================
#  PFAM / FGR / DCA / GSAF (your existing modules)
# ==============================================================

class PFAM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dim, dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim // 8, dim, 1),
            nn.Sigmoid()
        )
        self.freq_conv = nn.Sequential(
            nn.Conv3d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv3d(dim, dim, 1)
        )

    def forward(self, x):
        att = self.channel_att(x)
        high_freq = self.freq_conv(x)
        return high_freq * att


class DualCoordinateAttention(nn.Module):
    def __init__(self, dim, reduction=32):
        super().__init__()
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))
        self.conv1 = nn.Conv3d(dim, dim // reduction, 1)
        self.act = nn.ReLU(inplace=True)
        self.conv_d = nn.Conv3d(dim // reduction, dim, 1)
        self.conv_h = nn.Conv3d(dim // reduction, dim, 1)
        self.conv_w = nn.Conv3d(dim // reduction, dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, D, H, W = x.shape
        xd = self.pool_d(x)  # (B,C,D,1,1)
        xh = self.pool_h(x)  # (B,C,1,H,1)
        xw = self.pool_w(x)  # (B,C,1,1,W)

        xd = self.conv1(xd)
        xh = self.conv1(xh)
        xw = self.conv1(xw)

        xd = self.act(xd)
        xh = self.act(xh)
        xw = self.act(xw)

        ad = self.sigmoid(self.conv_d(xd))
        ah = self.sigmoid(self.conv_h(xh))
        aw = self.sigmoid(self.conv_w(xw))
        return x * ad * ah * aw


class GatedSpatialAdaptiveFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dca = DualCoordinateAttention(dim)
        self.pfam = PFAM(dim)

        # Router: pixel-wise weights for 2 experts
        self.router = nn.Conv3d(dim, 2, kernel_size=1, bias=True)

        # Forced residual: y = x + Δ, project Δ to 0 at init (stable lower bound)
        self.delta_proj = nn.Conv3d(dim, dim, kernel_size=1, bias=False)
        nn.init.constant_(self.delta_proj.weight, 0.0)

    def forward(self, x):
        out_dca = self.dca(x)
        out_pfam = self.pfam(x)

        logits = self.router(x)              # (B,2,D,H,W)
        weights = torch.softmax(logits, 1)   # (B,2,D,H,W)
        w_dca = weights[:, 0:1]
        w_pfam = weights[:, 1:2]

        fused = w_dca * out_dca + w_pfam * out_pfam
        delta = self.delta_proj(fused)
        return x + delta


class UNetResDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, nonlin_first: bool = False):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        # Decoder stage output channels in computation order (low -> high resolution).
        # HERC reads the final two entries instead of hard-coding plan-specific channels.
        stage_output_channels = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=encoder.conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedResidualBlocks(
                n_blocks = n_conv_per_stage[s-1],
                conv_op = encoder.conv_op,
                input_channels = 2 * input_features_skip,
                output_channels = input_features_skip,
                kernel_size = encoder.kernel_sizes[-(s + 1)],
                initial_stride = 1,
                conv_bias = encoder.conv_bias,
                norm_op = encoder.norm_op,
                norm_op_kwargs = encoder.norm_op_kwargs,
                dropout_op = encoder.dropout_op,
                dropout_op_kwargs = encoder.dropout_op_kwargs,
                nonlin = encoder.nonlin,
                nonlin_kwargs = encoder.nonlin_kwargs,
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))
            stage_output_channels.append(input_features_skip)

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.stage_output_channels = tuple(stage_output_channels)

    def forward(self, skips, return_features: bool = False):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        decoder_features = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            decoder_features.append(x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs

        if not return_features:
            return r

        if len(decoder_features) < 2:
            raise RuntimeError("HERC requires at least two decoder stages.")
        return r, {
            "decoder_coarse": decoder_features[-2],
            "decoder_full": decoder_features[-1],
            "decoder_all": decoder_features,
        }

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output


# ==============================================================
#  Soft MoE (Puigcerver et al., "From Sparse to Soft Mixtures of Experts", ICLR 2024)
#  Implemented following Algorithm 1 (dispatch + combine weights via per-slot parameters).
# ==============================================================
class SoftMoE(nn.Module):
    """
    Soft MoE routing (fully differentiable, no top-k, no token dropping).

    Input:  x  of shape (B, L, D)
    Output: y  of shape (B, L, D)

    Hyperparams:
      - num_experts (N): number of experts
      - slots_per_expert (P): number of slots per expert; total slots = N * P
      - ff_mult: expansion factor inside each expert MLP
    """
    def __init__(self,
                 d_model: int,
                 num_experts: int = 8,
                 slots_per_expert: int = 4,
                 ff_mult: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        assert num_experts >= 1 and slots_per_expert >= 1
        self.d_model = d_model
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.total_slots = num_experts * slots_per_expert

        # Per-slot parameters Φ ∈ R^{D x N x P}
        # logits[b,l,n,p] = <x[b,l,:], Φ[:,n,p]>
        self.phi = nn.Parameter(torch.randn(d_model, num_experts, slots_per_expert) * (d_model ** -0.5))

        # Experts f_i: typically MLPs
        experts = []
        hidden = d_model * ff_mult
        for _ in range(num_experts):
            experts.append(nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, d_model),
            ))
        self.experts = nn.ModuleList(experts)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape
        assert D == self.d_model, f"Expected last dim {self.d_model}, got {D}"

        # logits: (B, L, N, P)
        logits = torch.einsum("bld,dnp->blnp", x, self.phi)

        # Dispatch weights: softmax over tokens (per slot)
        dispatch = torch.softmax(logits, dim=1)

        # Combine weights: softmax over all slots (per token)
        combine = torch.softmax(logits.reshape(B, L, -1), dim=-1).reshape(B, L, self.num_experts, self.slots_per_expert)

        # Slot inputs: (B, N, P, D)
        slots_in = torch.einsum("bld,blnp->bnpd", x, dispatch)

        # Apply experts on their slots
        slots_out = []
        for n, expert in enumerate(self.experts):
            y_n = expert(slots_in[:, n, :, :])  # (B, P, D)
            slots_out.append(y_n)
        slots_out = torch.stack(slots_out, dim=1)  # (B, N, P, D)

        # Combine back to tokens: (B, L, D)
        y = torch.einsum("bnpd,blnp->bld", slots_out, combine)
        return self.dropout(y)



class SSDMoELayer(nn.Module):
    """
    Mamba2 + (optional) Attention + SoftMoE-FFN block.

    This replaces:
      - Mamba1 -> Mamba2 (SSD-based core)
      - Dense MLP -> SoftMoE (fully-differentiable MoE)
    """
    def __init__(self,
                 hidden_size: int,
                 d_state: int = 64,
                 d_conv: int = 4,
                 expand: int = 4,
                 mha_head: int = 8,
                 attn_dropout: float = 0.0,
                 # SoftMoE params
                 moe_num_experts: int = 8,
                 moe_slots_per_expert: int = 4,
                 moe_ff_mult: int = 4,
                 moe_dropout: float = 0.0):
        super().__init__()

        # Mamba-2 block (SSD). mamba-ssm==2.2.2 exposes it as `from mamba_ssm import Mamba2`.
        self.mamba = Mamba2(
            d_model=hidden_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=_choose_mamba2_headdim(hidden_size, min_heads=8),
        )

        # Keep attention branch (batch_first to match (B, L, C) tokens)
        self.mha = MHA(hidden_size, mha_head, dropout=attn_dropout, batch_first=True)

        self.mamba_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mha_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.moe_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        # SoftMoE replaces the dense MLP
        self.moe = SoftMoE(
            d_model=hidden_size,
            num_experts=moe_num_experts,
            slots_per_expert=moe_slots_per_expert,
            ff_mult=moe_ff_mult,
            dropout=moe_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        y = self.mha_norm(x)
        y, _ = self.mha(y, y, y, need_weights=False)
        x = x + y

        y = self.mamba_norm(x)
        # Ensure contiguous memory layout for Mamba2 (avoids stride alignment errors in fused causal_conv1d)
        y, orig_L = _mamba2_prepare(y, multiple=8)
        y = self.mamba(y)
        if y.shape[1] != orig_L:
            y = y[:, :orig_L, :]
        x = x + y

        y = self.moe_norm(x)
        y = self.moe(y)
        x = x + y

        return x


class SSDMoEBottleneck(nn.Module):
    """
    Stack of SSDMoELayer blocks.
    Kept the original API style: SSDMoEBottleneck(layer_num, hidden_size, ...).
    """
    def __init__(self,
                 layer_num: int,
                 hidden_size: int,
                 d_state: int = 64,
                 d_conv: int = 4,
                 expand: int = 4,
                 mha_head: int = 8,
                 attn_dropout: float = 0.0,
                 moe_num_experts: int = 8,
                 moe_slots_per_expert: int = 4,
                 moe_ff_mult: int = 4,
                 moe_dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            SSDMoELayer(
                hidden_size=hidden_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                mha_head=mha_head,
                attn_dropout=attn_dropout,
                moe_num_experts=moe_num_experts,
                moe_slots_per_expert=moe_slots_per_expert,
                moe_ff_mult=moe_ff_mult,
                moe_dropout=moe_dropout,
            )
            for _ in range(layer_num)
        ])
        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.last_norm(x)


class SSDMoEUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, None] = None,
                 stem_channels: Union[int, None] = None
                 ):
        super().__init__()
        self.num_classes = int(num_classes)

        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage * 2 ** i for i in range(n_stages)]
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        # [CHANGED] Encoder: Stage0-1 Multi-kernel parallel blocks, Stage2+ keep ResidualEncoder
        self.encoder = ShallowMultiKernelEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                               n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                               dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                               return_skips=True, disable_default_stem=False, stem_channels=stem_channels)

        # [NEW] Gated Spatial Adaptive Fusion (GSAF) modules for skip connections
        self.gsaf_modules = nn.ModuleList([
            GatedSpatialAdaptiveFusion(dim) for dim in features_per_stage[:-1]
        ])

        # [NEW] SSD-MoE bottleneck on the lowest resolution
        self.ssdmoe_bottleneck = SSDMoEBottleneck(12, features_per_stage[-1], d_state=64, d_conv=4, expand=4, mha_head=8)
        # Backward-compatible alias for older experiment scripts.
        self.mamba_mlp = self.ssdmoe_bottleneck

        # Decoder
        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x, return_features: bool = False):
        # 1. Encoder forward -> skips
        skips = self.encoder(x)

        # 2. Apply GSAF on skip features except bottleneck
        for i in range(len(skips) - 1):
            skips[i] = self.gsaf_modules[i](skips[i])

        # 3. Mamba bottleneck: flatten -> MambaMLP -> reshape
        middle_feature = skips[-1]
        B, C = middle_feature.shape[:2]
        img_dims = middle_feature.shape[2:]
        n_tokens = middle_feature.shape[2:].numel()
        middle_feature_flat = middle_feature.view(B, C, n_tokens).transpose(-1, -2).contiguous()
        middle_feature_flat, _orig_L0 = _mamba2_prepare(middle_feature_flat, multiple=8)
        # Mamba2 fused kernels require a well-aligned, contiguous memory layout
        middle_feature_flat = self.ssdmoe_bottleneck(middle_feature_flat)
        # if we padded length for Mamba2, slice back to original token count
        if middle_feature_flat.shape[1] != n_tokens:
            middle_feature_flat = middle_feature_flat[:, :n_tokens, :].contiguous()
        out = middle_feature_flat.transpose(-1, -2).view(B, C, *img_dims)
        skips[-1] = out

        # 4. Decoder. The optional feature return is strictly backward compatible:
        #    standard SSDMoE training/inference still receives exactly the original output.
        return self.decoder(skips, return_features=return_features)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


def get_ssdmoe_unet_from_plans(plans_manager: PlansManager,
                           dataset_json: dict,
                           configuration_manager: ConfigurationManager,
                           num_input_channels: int,
                           deep_supervision: bool = True):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = 'SSDMoEUNet'
    network_class = SSDMoEUNet
    kwargs = {
        'SSDMoEUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }

    conv_or_blocks_per_stage = {
        'n_conv_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }

    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                configuration_manager.unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))

    return model

# -----------------------------------------------------------------------------
# Backward-compatible aliases for older scripts and checkpoints
# -----------------------------------------------------------------------------
GatedSpatialFusion = GatedSpatialAdaptiveFusion
MultiKernelParallelResidualBlock = MultiKernelResidualBlock
StackedMultiKernelBlocks = StackedMultiKernelResidualBlocks
MultiKernelHybridEncoder = ShallowMultiKernelEncoder
MambaMLPLayer = SSDMoELayer
MambaMLP = SSDMoEBottleneck
MambaUNet = SSDMoEUNet
get_umamba_bot_from_plans = get_ssdmoe_unet_from_plans

__all__ = [
    "SSDMoEUNet",
    "SSDMoEBottleneck",
    "SSDMoELayer",
    "SoftMoE",
    "GatedSpatialAdaptiveFusion",
    "ShallowMultiKernelEncoder",
    "MultiKernelResidualBlock",
    "StackedMultiKernelResidualBlocks",
    "UNetResDecoder",
    "get_ssdmoe_unet_from_plans",
]
