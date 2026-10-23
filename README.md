# HERC — Heterogeneous Error Routing and Correction

Implementation of the correction module described in

> **HERC: Action-Constrained Correction of Residual Errors in Multi-Region Brain
> Tumor Segmentation** — Kaiyue Li, *Appl. Sci.* (submitted).

HERC attaches **one shared correction cell** to a frozen 3D segmenter. Residual
errors of the initial prediction are typed into four actions — false-negative
recovery, false-positive suppression, hierarchy correction and boundary
correction — and a bounded logit update is applied, bounded in a trust region and
re-diagnosed at the next step. The complete module adds **17,678 parameters**.

The repository is the SSDMoE-UNet project (nnU-Net v2 based) extended with HERC.
The baseline layout `evaluation/` + `umamba/nnunetv2/` is preserved.

## Status and scope

This repository contains the architecture, the training entry points and the
evaluation helpers of HERC. No trained checkpoint is distributed, and the
quantitative results reported in the manuscript come from a controlled simulation
of the evaluation protocol (see the simulation statement at the start of the
article) rather than from an executed training run. The parameter and compute
figures in `herc_static_audit.json` are counted on the model itself, not simulated.

## 1. Manuscript ⇄ code

| Manuscript (Section 3.2, Figures 2–3) | Code |
|---|---|
| shared correction cell, applied twice | `nets/herc/cell.py` → `HERCCell`, `HERCCorrector` |
| diagnosis at 64³ (severity, routes, step size) | `nets/herc/diagnosis.py` → `HERCDiagnosis` |
| four diagnostic cues, Equation 1 | `nets/herc/cues.py` |
| severity map `m_t`, route weights `w_t`, step size `eta_t` | keys `severity`, `route_weights`, `eta` |
| FN recovery expert, non-negative `b_s·σ(·)` | `FalseNegativeRecoveryExpert` |
| FP suppression expert, non-positive `−b_s·σ(·)` | `FalsePositiveSuppressionExpert` |
| hierarchy correction expert, boost/suppress | `HierarchyCorrectionExpert` |
| boundary correction expert at 128³, soft band | `BoundaryCorrectionExpert` |
| route targets from prediction and reference | `nets/herc/targets.py` → `build_route_targets` |
| Equation 5 objective | `training/loss/herc_loss.py` → `HERCCompositeLoss` |
| typed-error warm-up | `nets/herc/error_augmentation.py`, `nnUNetTrainerHERCWarmup` |
| frozen segmenter + corrector | `nets/HERCUNet.py` → `HERCUNet` |
| recovery, harm, precision (Section 3.5) | `evaluation/herc_metrics.py` |

The full old-to-new identifier table is in [NAMING.md](NAMING.md).

## 2. Pipeline position

```text
MRI (4 channels)
   -> frozen SSDMoE-UNet encoder -> GSAF skips -> SSD-MoE bottleneck -> decoder
   -> initial logits z_0  +  cached decoder features F^c @ 64^3, F^f @ 128^3
   -> HERC: diagnose -> type -> correct -> re-diagnose
   -> updated logits z_T -> baseline thresholds and export order
```

The backbone is never modified: `encoder.*`, `gsaf_modules.*`,
`ssdmoe_bottleneck.*` and `decoder.*` keep their names and shapes, so a
fold-specific SSDMoE-UNet checkpoint loads directly. The only new parameters live
under `herc.*`.

`SSDMoEUNet` gained one backward-compatible addition, used by HERC:

- `UNetResDecoder.forward(..., return_features=False)`
- `SSDMoEUNet.forward(..., return_features=False)`
- `decoder.stage_output_channels`

Standard SSDMoE calls receive exactly the original output.

## 3. Repository layout (HERC-specific files only)

```text
HERC/
├── README.md                     this file
├── NAMING.md                     naming convention and full rename table
├── HERC_DESIGN.md                design rationale
├── HERC_CODE_AUDIT.md            tensor, parameter, compute and memory audit
├── HERC_VERIFICATION.md          what was verified and what was not
├── HERC_FIXES_APPLIED.md         defect audit of the delivered package
├── HERC_INTEGRATION_DIFF.md      integration diff against the SSDMoE baseline
├── herc_static_audit.json        parameter/compute audit output (17,678 params)
├── legacy/
│   └── qa/verify_rename.py       checks that the rename is identifier-only
├── evaluation/
│   └── herc_metrics.py           recovery / harm / precision, hierarchy violations
└── umamba/nnunetv2/
    ├── nets/
    │   ├── HERCUNet.py           frozen SSDMoE-UNet + HERC
    │   ├── SSDMoEUNet.py         baseline (feature-return hook only)
    │   └── herc/
    │       ├── config.py         HERCConfig, ROUTE_NAMES, REGION_NAMES
    │       ├── blocks.py         ConvNormAct, ContextStem, ResidualOperator
    │       ├── cues.py           four cues, soft morphological band
    │       ├── targets.py        severity and route targets
    │       ├── diagnosis.py      HERCDiagnosis
    │       ├── experts.py        the four typed experts
    │       ├── cell.py           HERCCell, HERCCorrector
    │       └── error_augmentation.py  typed-error injection for the warm-up
    ├── training/
    │   ├── loss/herc_loss.py     HERCCompositeLoss
    │   └── nnUNetTrainer/nnUNetTrainerHERC.py
    ├── run/run_herc_static_audit.py
    └── tests/herc/test_herc_core.py, test_herc_loss.py
```

Every other file under `umamba/nnunetv2/` and `evaluation/` belongs to the
SSDMoE-UNet / nnU-Net v2 baseline and is intentionally untouched.

## 4. Region contract

The BraTS converter of the baseline produces nnU-Net **region** targets with
independent region heads:

```text
head 0 = WT,  head 1 = TC,  head 2 = ET        ET ⊆ TC ⊆ WT
=> P(WT) >= P(TC) >= P(ET)
```

HERC is region-aware by default; the trainer infers the WT/TC/ET head indices from
`label_manager.foreground_regions` instead of assuming channel order. The third
expert is therefore a hierarchy correction expert, not a generic class-confusion
module. An atomic multiclass fallback is retained for non-region datasets.

Route order, shared by diagnosis, targets and metrics:

```text
0 = fn (false-negative recovery)      1 = fp (false-positive suppression)
2 = hierarchy (correction)            3 = boundary (correction)
```

## 5. Defaults reported in the manuscript

| Setting | Value | `HERCConfig` field |
|---|---|---|
| diagnostic resolution / channels | 64³ / 32 | `diagnosis_feature_channels`, `diagnosis_hidden_channels` |
| expert width | 24 | `expert_hidden_channels` |
| semantic bound `b_s` | 2.0 | `semantic_residual_bound` |
| boundary bound `b_b` | 1.5 | `boundary_residual_bound` |
| trust-region bound `b_g` | 2.0 | `global_update_bound` |
| step size `eta` range | [0.05, 1.0] | `min_step_size`, `max_step_size` |
| train / inference steps | 2 / ≤3 | `train_steps`, `max_inference_steps` |
| band radius | 1 | `boundary_radius` |
| loss weights `λ_s, λ_r, λ_b, λ_m, λ_safe, λ_h` | 0.20, 0.10, 0.10, 0.20, 0.10, 0.05 | `HERCCompositeLoss` |

## 6. Training phases

| Phase | Trainer | Frozen | Steps | Epochs |
|---|---|---|---|---|
| typed-error warm-up | `nnUNetTrainerHERCWarmup` | backbone | 1 | 50 |
| natural residual errors (primary) | `nnUNetTrainerHERC` | backbone | 2 | 200 |
| joint fine-tuning (optional) | `nnUNetTrainerHERCJoint` | late decoder + heads | 2 | 100 |

Warm-up errors are not image noise: the augmentation injects FN, FP, hierarchy
violation and boundary displacement into the logits of the frozen segmenter.

```bash
nnUNetv2_train DATASET_ID 3d_fullres 0 \
  -tr nnUNetTrainerHERCWarmup \
  -pretrained_weights /path/to/SSDMoE/checkpoint_final.pth

nnUNetv2_train DATASET_ID 3d_fullres 0 \
  -tr nnUNetTrainerHERC \
  -pretrained_weights /path/to/warmup/checkpoint_final.pth

HERC_JOINT_LEVEL=2 nnUNetv2_train DATASET_ID 3d_fullres 0 \
  -tr nnUNetTrainerHERCJoint \
  -pretrained_weights /path/to/natural/checkpoint_final.pth
```

## 7. Environment variables

All settings are reachable through `HERC_*` variables:

```text
HERC_EPOCHS                    HERC_LR                       HERC_WEIGHT_DECAY
HERC_TRAIN_STEPS               HERC_MAX_INFERENCE_STEPS      HERC_ADAPTIVE_STOP
HERC_STOP_THRESHOLD            HERC_STOP_TOPK_FRACTION
HERC_LAMBDA_SEVERITY           HERC_LAMBDA_ROUTE             HERC_LAMBDA_BOUNDARY
HERC_LAMBDA_MONOTONIC          HERC_LAMBDA_SAFETY            HERC_LAMBDA_HIERARCHY
HERC_LAMBDA_ANCHOR             HERC_LAMBDA_COARSE            HERC_JOINT_LEVEL
HERC_JOINT_DECODER_STAGES      HERC_HEAD_LR_RATIO            HERC_DECODER_LR_RATIO
HERC_GSAF_LR_RATIO             HERC_TUNE_SSDMOE
HERC_SSDMOE_LR_RATIO           HERC_SSDMOE_ROUTING_LR_RATIO
```

## 8. Checkpoint compatibility

`run/load_pretrained_weights.py` loads a baseline SSDMoE checkpoint into
`HERCUNet`: all shared keys are shape-checked, and the absent `herc.*` keys are
allowed. Note that the correction prefix changed during the rename, so a
checkpoint written by the pre-rename code needs one remapping step:

```python
import torch

state = torch.load('checkpoint_final.pth', map_location='cpu')
state['network_weights'] = {
    k.replace('fail2correct.', 'herc.', 1): v
    for k, v in state['network_weights'].items()
}
torch.save(state, 'checkpoint_final.herc.pth')
```

## 9. Verification

`HERC_CODE_AUDIT.md` reports the parameter, compute and memory audit;
`HERC_VERIFICATION.md` lists what passed and what still requires the real GPU
environment; `herc_static_audit.json` is the machine-readable audit output:

```bash
python -m nnunetv2.run.run_herc_static_audit --patch 128 --steps 2 --json herc_static_audit.json
python -m nnunetv2.tests.herc.test_herc_core
python -m nnunetv2.tests.herc.test_herc_loss
```

The full SSDMoE forward requires `mamba_ssm`, `dynamic-network-architectures`,
`batchgenerators` and `acvl-utils`, which are not part of this package.
