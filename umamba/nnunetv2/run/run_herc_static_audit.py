"""Static resource audit of the HERC correction cell.

Counts the parameters of one configured cell and estimates its convolution cost
from an analytical MAC model at a fixed patch size.  The numbers reproduce the
17,678-parameter budget reported for the shared module in the manuscript.

Usage:
    python -m nnunetv2.run.run_herc_static_audit --patch 128 --steps 2
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from nnunetv2.nets.herc.cell import HERCCorrector
from nnunetv2.nets.herc.config import HERCConfig


def conv_macs(cin: int, cout: int, k: int, voxels: int, groups: int = 1) -> int:
    """Multiply-accumulate operations of one 3D convolution."""
    return voxels * cout * (cin // groups) * (k ** 3)


def analytical_macs(
    patch: int = 128,
    coarse_channels: int = 64,
    full_channels: int = 32,
    steps: int = 2,
) -> dict:
    """Analytical convolution cost of one cell applied ``steps`` times."""
    coarse = patch // 2
    v64 = coarse ** 3
    v128 = patch ** 3

    # Static feature projections are cached once for all correction steps.
    static = conv_macs(coarse_channels, 32, 1, v64) + conv_macs(full_channels, 8, 1, v128)

    dynamic = 0
    # Diagnosis context stem: the region contract has 39 input channels.
    dynamic += conv_macs(39, 32, 1, v64)
    dynamic += conv_macs(32, 32, 3, v64, groups=32)
    dynamic += conv_macs(32, 32, 1, v64)
    dynamic += conv_macs(32, 32, 3, v64, groups=32)
    dynamic += conv_macs(32, 1, 1, v64)
    dynamic += conv_macs(32, 4, 1, v64)
    dynamic += 32 * 16 + 16

    def expert(out_channels: int) -> int:
        return (
            conv_macs(35, 24, 1, v64)
            + conv_macs(24, 24, 3, v64, groups=24)
            + conv_macs(24, 24, 1, v64)
            + conv_macs(24, 24, 3, v64, groups=24)
            + conv_macs(24, out_channels, 1, v64)
        )

    dynamic += expert(3) + expert(3) + expert(4)

    # Boundary context stem and boundary operator.
    dynamic += conv_macs(16, 8, 1, v128)
    dynamic += conv_macs(8, 8, 3, v128, groups=8)
    dynamic += conv_macs(8, 8, 1, v128)
    dynamic += conv_macs(8, 8, 3, v128, groups=8)

    dynamic += conv_macs(8, 8, 1, v128)
    dynamic += conv_macs(8, 8, 3, v128, groups=8)
    dynamic += conv_macs(8, 8, 1, v128)
    dynamic += conv_macs(8, 8, 3, v128, groups=8)
    dynamic += conv_macs(8, 6, 1, v128)

    total = static + steps * dynamic
    return {
        'static_gmac': static / 1e9,
        'dynamic_per_step_gmac': dynamic / 1e9,
        'total_gmac': total / 1e9,
        'strict_gflops_2_per_mac': 2 * total / 1e9,
    }


def tensor_footprint(patch: int = 128) -> list:
    """Peak activation footprint per tensor of the region-mode cell."""
    coarse = patch // 2
    specs = [
        ('MRI input', 4, patch),
        ('segmenter logits', 3, patch),
        ('decoder_coarse', 64, coarse),
        ('decoder_full', 32, patch),
        ('diagnosis projected', 32, coarse),
        ('diagnosis concat', 39, coarse),
        ('diagnosis context', 32, coarse),
        ('severity', 1, coarse),
        ('route weights', 4, coarse),
        ('expert input', 35, coarse),
        ('semantic residual', 3, coarse),
        ('semantic residual upsampled', 3, patch),
        ('boundary projected', 8, patch),
        ('boundary probabilities', 3, patch),
        ('soft morphological band', 3, patch),
        ('boundary route', 1, patch),
        ('MRI gradient', 1, patch),
        ('boundary concat', 16, patch),
        ('boundary hidden', 8, patch),
        ('boundary raw out+in', 6, patch),
        ('boundary residual', 3, patch),
        ('final delta', 3, patch),
    ]
    out = []
    for name, channels, size in specs:
        elements = channels * size ** 3
        out.append({
            'name': name,
            'shape': [1, channels, size, size, size],
            'elements': elements,
            'fp16_mib': elements * 2 / 2 ** 20,
            'fp32_mib': elements * 4 / 2 ** 20,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patch', type=int, default=128)
    parser.add_argument('--coarse_channels', type=int, default=64)
    parser.add_argument('--full_channels', type=int, default=32)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--json', type=str, default=None)
    args = parser.parse_args()

    cfg = HERCConfig(num_outputs=3, has_regions=True, train_steps=args.steps)
    model = HERCCorrector(args.coarse_channels, args.full_channels, cfg)
    breakdown = {
        name: sum(p.numel() for p in module.parameters())
        for name, module in model.cell.named_children()
    }
    result = {
        'config': asdict(cfg),
        'parameters_total': sum(p.numel() for p in model.parameters()),
        'parameter_breakdown': breakdown,
        'analytical_convolution_complexity': analytical_macs(
            args.patch, args.coarse_channels, args.full_channels, args.steps
        ),
        'tensor_footprint': tensor_footprint(args.patch),
        'notes': [
            'GMAC counts convolution and linear multiply-accumulate operations; '
            'interpolation, pooling, activations and elementwise ops are omitted.',
            'strict_gflops_2_per_mac reports the convention where one multiply and '
            'one add count as two FLOPs.',
            'For comparison with the reported SSDMoE GFLOPs, rerun the exact same '
            'profiler that was used for the baseline.',
        ],
    }
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2)


if __name__ == '__main__':
    main()
