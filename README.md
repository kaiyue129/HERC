# HERC — Heterogeneous Error Routing and Correction

Implementation of the correction module described in

> **HERC: Action-Constrained Residual Correction for Multi-Region Brain Tumor
> Segmentation** — Kaiyue Li, *Appl. Sci.* (submitted).

HERC attaches **one shared correction cell** to a frozen 3D segmenter. Residual
errors of the initial prediction are typed into four actions — false-negative
recovery, false-positive suppression, hierarchy correction and boundary
correction — and a bounded logit update is applied and re-diagnosed at the next
step. The complete module adds **17,678 parameters**.

The repository is the SSDMoE-UNet project (nnU-Net v2 based) extended with HERC.
The baseline layout `evaluation/` + `umamba/nnunetv2/` is preserved.

## Status and scope

This repository is the code accompanying the paper: the correction module, the
training entry points, the analytical cost audit and the evaluation helpers of
HERC.

The primary evaluation used 1251 cases from BraTS 2023. The SSDMoE-UNet baseline
and the HERC correction stage were trained exclusively on that release, with
patient-wise nnU-Net-style five-fold training repeated over three independent
random seeds, and 500 epochs for every fold-specific run. Within one seed the five
fold-specific baseline–HERC pipelines are combined through the nnU-Net-style
prediction procedure, which yields one ensemble-level prediction per case; the
reported values are the mean and standard deviation over the three seeds. BraTS
2021 and BraTS 2018 were evaluated with the same trained models, without
retraining, fine-tuning or release-specific fitting. HERC was also fitted
separately for the other three initial segmenters (SegResNet, LGA-Net, Ga-U-Net).

The BraTS images and labels are not redistributed here, under the challenge
data-use conditions. Patient-wise fold assignments, evaluation outputs and
configuration files are available from the author on reasonable request.

Headline numbers from the manuscript: on BraTS 2023, mean full-volume Dice
improves from 90.40% to 91.80% and lesion-wise Dice from 85.75% to 87.79%, which
is +0.58 pp over a magnitude-matched unsigned corrector and +0.59 pp over an
untyped mixture at equal capacity.

## 1. Manuscript ⇄ code

| Manuscript (Section 3.2, Figures 2–3) | Code |
|---|---|
| shared correction cell, reused for two steps | `nets/herc/cell.py` → `HERCCell`, `HERCCorrector` |
| diagnosis at 64³ (severity, routes, step size) | `nets/herc/diagnosis.py` → `HERCDiagnosis` |
| four diagnostic cues, Equation 1 | `nets/herc/cues.py` |
| severity map `m_t`, route weights `w_t`, step size `eta_t` | keys `severity`, `route_weights`, `eta` |
| FN recovery expert, non-negative `b_s·σ(·)` | `FalseNegativeRecoveryExpert` |
| FP suppression expert, non-positive `−b_s·σ(·)` | `FalsePositiveSuppressionExpert` |
| hierarchy correction expert, boost/suppress | `HierarchyCorrectionExpert` |
| boundary correction expert at 128³, soft band | `BoundaryCorrectionExpert` |
| route targets from prediction and reference | `nets/herc/targets.py` → `build_route_targets` |
| Equation 5 objective | `training/loss/herc_loss.py` → `HERCCompositeLoss` |
| frozen segmenter + corrector | `nets/HERCUNet.py` → `HERCUNet` |
| recovery, harm, precision (Section 3.5) | `evaluation/herc_metrics.py` |

Identifiers follow the manuscript: the method is `HERC`, tensors use the paper
symbols (`severity`, `route_weights`, `eta`, `r_fn`, `r_h`), and the four experts
are named after the action they perform. See
[Code conventions](#10-code-conventions).

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
├── DESIGN.md                     design rationale: why four typed actions
├── herc_static_audit.json        parameter/compute audit output (17,678 params)
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
    │       └── error_augmentation.py  typed-error injection helper
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

HERC is region-aware: the trainer infers the WT/TC/ET head indices from
`label_manager.foreground_regions` instead of assuming channel order, so the third
expert is a hierarchy correction expert rather than a class-confusion module.

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
| correction steps | 2 | `train_steps` |
| band radius | 1 | `boundary_radius` |
| loss weights `λ_s, λ_r, λ_b, λ_m, λ_safe, λ_h` | 0.20, 0.10, 0.10, 0.20, 0.10, 0.05 | `HERCCompositeLoss` |

## 6. Training and inference

As reported in Section 3.3: four-channel `128³` patches, batch size one, 250 patch
draws per epoch, 500 epochs for every fold-specific run of each of the three seeds,
which is 125,000 optimizer steps per run. SGD with Nesterov momentum 0.99, initial
learning rate `1e-2`, weight decay `3e-5`, polynomial decay exponent 0.9, mixed
precision, gradient clipping at L2 norm 1.0, and the region-based nnU-Net
augmentation policy unmodified. The held-out fold is removed first and the
remaining 80% of patients is split 4:1 by patient into correction training and
inner validation; the retained checkpoint maximises exponential-moving-average
foreground Dice on that inner split. Two correction steps run inside each
fold-specific pipeline, and at inference the five fold-specific pipelines of a seed
are combined through the nnU-Net-style prediction procedure.

```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD \
  -tr nnUNetTrainerHERC \
  -pretrained_weights /path/to/SSDMoE/checkpoint_final.pth
```

## 7. Environment variables

The settings of Section 5 and 6 are reachable through `HERC_*` variables:

```text
HERC_EPOCHS                    HERC_LR                       HERC_WEIGHT_DECAY
HERC_TRAIN_STEPS               HERC_MAX_INFERENCE_STEPS      HERC_ADAPTIVE_STOP
HERC_STOP_THRESHOLD            HERC_STOP_TOPK_FRACTION
HERC_LAMBDA_SEVERITY           HERC_LAMBDA_ROUTE             HERC_LAMBDA_BOUNDARY
HERC_LAMBDA_MONOTONIC          HERC_LAMBDA_SAFETY            HERC_LAMBDA_HIERARCHY
```

## 8. Checkpoint compatibility

`run/load_pretrained_weights.py` loads a fold-specific baseline checkpoint into
`HERCUNet`: every shared key is shape-checked, and the absent `herc.*` keys are
accepted, so the correction stage starts from the trained backbone. The backbone
keys themselves are unchanged (`encoder.*`, `gsaf_modules.*`,
`ssdmoe_bottleneck.*`, `decoder.*`).

## 9. Cost and resources

Parameter budget of the shared cell, counted on the region-mode model
(`herc_static_audit.json`):

| Component | Parameters |
|---|---:|
| diagnosis | 7,078 |
| false-negative expert | 2,979 |
| false-positive expert | 2,979 |
| hierarchy correction expert | 3,004 |
| boundary correction expert | 1,638 |
| **total, added by HERC** | **17,678** |

The corrector therefore adds 0.0177 M parameters (17,678) to the frozen
segmenter, so the backbone stays the parameter count of the baseline.

Convolution-dominated cost at `128³`, batch one, with 64/32 decoder taps
(MACs only; interpolation, pooling, activations and morphology excluded):

| Setting | GMAC |
|---|---:|
| static projections, once | 1.074 |
| dynamic cell, per step | 5.870 |
| T = 1 | 6.944 |
| **T = 2 (primary)** | **12.814** |
| T = 3 | 18.684 |

Under the strict two-operation convention T = 2 is about 25.63 GFLOPs. These are
analytical convolution counts of the released cell, not profiler outputs.

## 10. Code conventions

- the method is `HERC`; tensor keys use the paper symbols;
- modules are 3D by construction, so class names carry no topology suffix;
- experts are named after the action they perform (`BoundaryCorrectionExpert`);
- the correction sub-module is reachable as `model.herc` and owns `herc.*` weights;
- one module docstring per file names the section of the manuscript it implements.

## 11. Verification

`herc_static_audit.json` is the machine-readable parameter and compute audit; the
two shipped tests and the audit script can be rerun with:

```bash
python -m nnunetv2.run.run_herc_static_audit --patch 128 --steps 2 --json herc_static_audit.json
python -m nnunetv2.tests.herc.test_herc_core
python -m nnunetv2.tests.herc.test_herc_loss
```

Checked on the released tree: every Python module parses and its intra-project
imports resolve; every configuration and loss default is the value reported in
Section 5; the layer construction reproduces the parameter table of Section 9.
Running an end-to-end forward and backward pass requires `mamba_ssm`,
`dynamic-network-architectures`, `batchgenerators` and `acvl-utils` plus a CUDA
device, none of which are bundled here; see `umamba/setup.py` for the package
itself.
