"""Composite-loss test of the HERC correction objective.

Run with:
    python -m nnunetv2.tests.herc.test_herc_loss
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch import nn

from nnunetv2.nets.herc.cell import HERCCorrector
from nnunetv2.nets.herc.config import HERCConfig
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.herc_loss import HERCCompositeLoss


class RegionPrimitive(nn.Module):
    """Minimal region primitive standing in for the nnU-Net Dice + BCE loss."""

    def forward(self, logits, target):
        return F.binary_cross_entropy_with_logits(logits, target.float())


def main() -> None:
    torch.manual_seed(1)
    cfg = HERCConfig(num_outputs=3, has_regions=True, train_steps=2)
    model = HERCCorrector(64, 32, cfg).train()

    image = torch.randn(1, 4, 32, 32, 32)
    coarse_feature = torch.randn(1, 64, 16, 16, 16)
    full_feature = torch.randn(1, 32, 32, 32, 32)
    initial_logits = torch.randn(1, 3, 32, 32, 32)
    correction = model(image, coarse_feature, full_feature, initial_logits)

    low = F.interpolate(initial_logits, size=(16, 16, 16), mode='trilinear', align_corners=False)
    baseline_outputs = [initial_logits, low]
    corrected_outputs = [correction['final_logits'], low]

    target_hi = (torch.rand(1, 3, 32, 32, 32) > 0.7).float()
    # Enforce WT >= TC >= ET for the semantics of the test.
    target_hi[:, 1] *= target_hi[:, 0]
    target_hi[:, 2] *= target_hi[:, 1]
    target_low = F.interpolate(target_hi, size=(16, 16, 16), mode='nearest')
    targets = [target_hi, target_low]

    deep_supervision = DeepSupervisionWrapper(RegionPrimitive(), [2 / 3, 1 / 3])
    loss_fn = HERCCompositeLoss(deep_supervision, has_regions=True)
    pack = {
        'seg_outputs': corrected_outputs,
        'initial_outputs': baseline_outputs,
        'baseline_outputs': baseline_outputs,
        'herc': correction,
    }
    loss = loss_fn(pack, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert 'severity' in loss_fn.last_components
    print('HERC composite-loss test passed:', float(loss.detach()))


if __name__ == '__main__':
    main()
