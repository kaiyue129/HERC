"""Correction-level metrics of the HERC evaluation (Section 3.5).

Correction is counted on binary region elements, because one voxel can belong to
several nested regions.  ``R`` counts initially erroneous elements that became
correct and ``H`` counts initially correct elements that became incorrect.
"""

from __future__ import annotations

import numpy as np

#: Region order of the BraTS region heads.
REGION_NAMES = ('WT', 'TC', 'ET')

#: Route order of the four typed actions (Figure 3).
ROUTE_NAMES = ('fn', 'fp', 'hierarchy', 'boundary')


def region_correction_metrics(coarse_regions, corrected_regions, target_regions) -> dict:
    """Element-level recovery, harm and precision of one correction.

    Args:
        coarse_regions: Boolean/int array ``[R, D, H, W]`` or ``[B, R, D, H, W]``
            holding the initial prediction.
        corrected_regions: Same layout, holding the corrected prediction.
        target_regions: Same layout, holding the reference.

    Returns:
        Pooled and per-region recovery/harm rates, the correction precision and
        the fraction of changed elements.
    """
    coarse = np.asarray(coarse_regions).astype(bool)
    corrected = np.asarray(corrected_regions).astype(bool)
    target = np.asarray(target_regions).astype(bool)

    coarse_correct = coarse == target
    corrected_correct = corrected == target
    original_wrong = ~coarse_correct
    original_right = coarse_correct
    changed = coarse != corrected

    recovered = original_wrong & corrected_correct
    harmed = original_right & (~corrected_correct)
    helpful = changed & original_wrong & corrected_correct

    metrics = {
        'error_recovery_rate': float(recovered.sum() / max(1, original_wrong.sum())),
        'harm_rate': float(harmed.sum() / max(1, original_right.sum())),
        'correction_precision': float(helpful.sum() / max(1, changed.sum())),
        'changed_element_fraction': float(changed.mean()),
    }

    axis_offset = 1 if coarse.ndim == 5 else 0
    for index, name in enumerate(REGION_NAMES):
        selection = [slice(None)] * coarse.ndim
        selection[axis_offset] = index
        selection = tuple(selection)
        wrong = original_wrong[selection]
        right = original_right[selection]
        recovered_region = recovered[selection]
        harmed_region = harmed[selection]
        metrics[f'{name}_recovery_rate'] = float(recovered_region.sum() / max(1, wrong.sum()))
        metrics[f'{name}_harm_rate'] = float(harmed_region.sum() / max(1, right.sum()))

    return metrics


def hierarchy_violation_rate(prob_regions, threshold: float = 0.0) -> float:
    """Fraction of elements whose predicted hierarchy violates WT >= TC >= ET.

    Args:
        prob_regions: Region probabilities ``[3, D, H, W]`` or ``[B, 3, D, H, W]``
            in WT, TC, ET order.
        threshold: Minimum excess probability that counts as a violation.
    """
    prob = np.asarray(prob_regions)
    axis = 1 if prob.ndim == 5 else 0
    wt = np.take(prob, 0, axis=axis)
    tc = np.take(prob, 1, axis=axis)
    et = np.take(prob, 2, axis=axis)
    violation = (tc - wt > threshold) | (et - tc > threshold)
    return float(violation.mean())
