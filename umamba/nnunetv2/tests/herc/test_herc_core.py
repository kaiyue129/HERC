"""Core shape and sign-invariant tests of the HERC correction cell.

Run with:
    python -m nnunetv2.tests.herc.test_herc_core
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from nnunetv2.nets.herc.cell import HERCCorrector
from nnunetv2.nets.herc.config import HERCConfig
from nnunetv2.nets.herc.cues import hierarchy_violation_cue
from nnunetv2.nets.herc.error_augmentation import inject_typed_error
from nnunetv2.nets.herc.targets import build_route_targets


def make_nested_target(size: int = 32) -> torch.Tensor:
    """Build a nested WT >= TC >= ET reference."""
    target = torch.zeros(1, 3, size, size, size)
    target[:, 0, 5:27, 5:27, 5:27] = 1  # WT
    target[:, 1, 9:23, 9:23, 9:23] = 1  # TC
    target[:, 2, 12:20, 12:20, 12:20] = 1  # ET
    return target


def main() -> None:
    torch.manual_seed(0)
    cfg = HERCConfig(
        num_outputs=3,
        has_regions=True,
        train_steps=2,
        adaptive_stopping=False,
    )
    model = HERCCorrector(
        coarse_decoder_channels=64,
        full_decoder_channels=32,
        cfg=cfg,
    )
    model.train()

    image = torch.randn(1, 4, 32, 32, 32)
    coarse = torch.randn(1, 64, 16, 16, 16)
    full = torch.randn(1, 32, 32, 32, 32)
    logits = torch.randn(1, 3, 32, 32, 32)

    out = model(image, coarse, full, logits)
    assert out['final_logits'].shape == logits.shape
    assert len(out['step_states']) == 2

    state = out['step_states'][0]
    assert state['severity'].shape == (1, 1, 16, 16, 16)
    assert state['route_weights'].shape == (1, 4, 16, 16, 16)
    assert state['route_weights'].sum(dim=1).sub(1.0).abs().max() < 1e-5
    assert state['r_fn'].shape == (1, 3, 16, 16, 16)
    assert state['r_boundary'].shape == logits.shape

    # Sign restrictions of the two signed semantic experts.
    assert (state['r_fn'] >= 0).all()
    assert (state['r_fp'] <= 0).all()

    # The severity map and the step size stay in their documented ranges.
    assert state['severity'].min() >= 0 and state['severity'].max() <= 1
    assert state['eta'].min() >= cfg.min_step_size and state['eta'].max() <= cfg.max_step_size

    target = make_nested_target(32)
    route_target = build_route_targets(logits, target, has_regions=True, boundary_radius=1)
    assert route_target['severity'].shape == (1, 1, 32, 32, 32)
    assert route_target['route'].shape == (1, 4, 32, 32, 32)

    # Typed error injection preserves the tensor shape and reports its type.
    perturbed, kind = inject_typed_error(logits, target, True, mode='hierarchy')
    assert perturbed.shape == logits.shape
    assert kind == 'hierarchy'

    # An explicit hierarchy violation has to be detected by the cue.
    prob = torch.zeros(1, 3, 4, 4, 4)
    prob[:, 0] = 0.2  # WT
    prob[:, 1] = 0.8  # TC > WT
    prob[:, 2] = 0.9  # ET > TC
    assert hierarchy_violation_cue(prob).mean() > 0

    print('HERC region-mode core tests passed.')


if __name__ == '__main__':
    main()
