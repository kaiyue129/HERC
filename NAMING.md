# Naming convention

This file fixes the naming scheme of the code base and records every rename that
was applied to align it with the manuscript. The rule of thumb is that a reader of
the paper can find the corresponding symbol by its paper name.

## 1. Why the rename

The pre-rename code called the method **Fail2Correct** (`fail2correct`, `f2c`,
`F2C_`), described errors as **failures** (`FailureDiagnosis3D`,
`induce_typed_failure`) and used inconsistent topological suffixes
(`FailureDiagnosis3D`, `ConvNormAct3d`). The manuscript calls the method
**HERC** (Heterogeneous Error Routing and Correction), calls the residual
structure an **error**, and types it into four **actions** — false-negative
recovery, false-positive suppression, hierarchy correction and boundary
correction. The code now uses exactly that vocabulary.

## 2. Rules

### 2.1 Project and method names

| Item | Rule | Example |
|---|---|---|
| project root | upper-case acronym | `HERC/` |
| python package | lower-case, short | `nets/herc/` |
| method acronym in identifiers | upper-case inside CapWords | `HERCConfig`, `HERCCell`, `HERCUNet` |
| baseline name | keep the cited name unchanged | `SSDMoEUNet`, `nnUNetTrainerSSDMoEUNet` |

`SSDMoE-UNet` is the *initial segmenter* cited as reference [40]; it is not
renamed. HERC is the name of the added module, so the corrected network is
`HERCUNet` and the backbone-specific part is expressed by inheritance, not by a
longer class name.

### 2.2 Files and modules

| Item | Rule | Example |
|---|---|---|
| module file | lower-case noun, singular, paper term | `diagnosis.py`, `cues.py`, `experts.py` |
| nnU-Net net module | CapWords, mirrors the class | `HERCUNet.py` |
| trainer module | `nnUNetTrainer<Method>.py` | `nnUNetTrainerHERC.py` |
| runner module | `run_<thing>.py` | `run_herc_static_audit.py` |
| test module | `test_<module>.py` under `tests/herc/` | `test_herc_core.py` |
| documentation | `HERC_<TOPIC>.md` | `HERC_CODE_AUDIT.md` |
| data artifact | lower-case, produced by a script of the same stem | `herc_static_audit.json` |
| historical material | under `legacy/`, pre-rename name kept in the stem | `legacy/FAIL2CORRECT_pre_rename.patch` |

### 2.3 Classes

| Item | Rule | Example |
|---|---|---|
| class | CapWords, noun phrase | `HERCDiagnosis` |
| **no topology suffix** | every module here is 3D by construction | `ConvNormAct`, not `ConvNormAct3d` |
| expert classes | `<Action><Verb>Expert` | `FalseNegativeRecoveryExpert`, `BoundaryCorrectionExpert` |
| abbreviations | spelled out except WT/TC/ET and HERC | `FalsePositiveSuppressionExpert`, not `FPSuppressionExpert3D` |
| private helper | leading underscore | `_normalise_route` |

### 2.4 Functions and variables

| Item | Rule | Example |
|---|---|---|
| function | `snake_case`, verb first | `build_route_targets`, `inject_typed_error` |
| cue function | `<quantity>_cue` | `entropy_cue`, `hierarchy_violation_cue` |
| boolean | `is_`/`has_`/`use_` prefix | `has_regions`, `use_checkpoint` |
| tensor keys | short `snake_case` mirroring the paper symbol | `severity`, `route_weights`, `eta`, `delta`, `r_fn`, `r_h` |
| environment variable | `HERC_<SETTING>` | `HERC_TRAIN_STEPS`, `HERC_LAMBDA_ROUTE` |
| module attribute of the corrector | `herc` (never `fail2correct`) | `model.herc`, `model.herc_config` |

Route and region letters are kept short because they appear inside tensor names:
`fn`, `fp`, `h` (hierarchy), `b` (boundary); `WT`, `TC`, `ET`.

### 2.5 Formatting

* one module docstring per file, stating which section of the manuscript it
  implements;
* `from __future__ import annotations` in every module with type hints;
* 4-space indentation, single quotes, lines ≤ 100 characters;
* two blank lines between top-level definitions, one between methods;
* Google-style docstrings for public classes and functions, `Args:`/`Returns:`
  only where the names are not self-explanatory;
* no trailing whitespace, single trailing newline.

## 3. Complete rename table

### 3.1 Files and directories

| Before | After |
|---|---|
| `SSDMoE-Fail2Correct-full/` | `HERC/` |
| `umamba/nnunetv2/nets/fail2correct/` | `umamba/nnunetv2/nets/herc/` |
| `nets/fail2correct/signals.py` | `nets/herc/cues.py` |
| `nets/fail2correct/failure_augmentation.py` | `nets/herc/error_augmentation.py` |
| `nets/SSDMoEFail2Correct.py` | `nets/HERCUNet.py` |
| `training/loss/fail2correct_losses.py` | `training/loss/herc_loss.py` |
| `training/nnUNetTrainer/nnUNetTrainerSSDMoEFail2Correct.py` | `training/nnUNetTrainer/nnUNetTrainerHERC.py` |
| `run/run_fail2correct_static_audit.py` | `run/run_herc_static_audit.py` |
| `evaluation/fail2correct_metrics.py` | `evaluation/herc_metrics.py` |
| `tests/fail2correct/test_fail2correct_core.py` | `tests/herc/test_herc_core.py` |
| `tests/fail2correct/test_fail2correct_loss.py` | `tests/herc/test_herc_loss.py` |
| `FAIL2CORRECT_README.md` | `README.md` |
| `FAIL2CORRECT_DESIGN.md` | `HERC_DESIGN.md` |
| `EXPERIMENT_CODE_AUDIT.md` | `HERC_CODE_AUDIT.md` |
| `VERIFICATION.md` | `HERC_VERIFICATION.md` |
| `FIXES_APPLIED.md` | `HERC_FIXES_APPLIED.md` |
| `INTEGRATION_DIFF_SUMMARY.md` | `HERC_INTEGRATION_DIFF.md` |
| `fail2correct_static_audit.json` | `herc_static_audit.json` |
| `FAIL2CORRECT.patch` | `legacy/FAIL2CORRECT_pre_rename.patch` |
| `PROJECT_TREE.txt` | regenerated |

Unchanged on purpose: `LICENSE`, `.gitignore`, `umamba/setup.py`,
`nets/SSDMoEUNet.py`, `training/nnUNetTrainer/nnUNetTrainerSSDMoEUNet.py`, all
nnU-Net v2 baseline modules, and the non-BraTS evaluation helpers
(`SurfaceDice.py`, `compute_cell_metric.py`, `abdomen_*`, `endoscopy_*`).

### 3.2 Classes

| Before | After |
|---|---|
| `Fail2CorrectConfig` | `HERCConfig` |
| `Fail2CorrectCell` | `HERCCell` |
| `IterativeFail2Correct` | `HERCCorrector` |
| `FailureDiagnosis3D` | `HERCDiagnosis` |
| `FNRecoveryExpert3D` | `FalseNegativeRecoveryExpert` |
| `FPSuppressionExpert3D` | `FalsePositiveSuppressionExpert` |
| `ForegroundConfusionExpert3D` | `ForegroundConfusionExpert` |
| `HierarchyConsistencyExpert3D` | `HierarchyCorrectionExpert` |
| `FullResolutionBoundaryExpert3D` | `BoundaryCorrectionExpert` |
| `Fail2CorrectCompositeLoss` | `HERCCompositeLoss` |
| `SSDMoEFail2CorrectUNet` | `HERCUNet` |
| `nnUNetTrainerSSDMoEFail2Correct` | `nnUNetTrainerHERC` |
| `nnUNetTrainerSSDMoEFail2CorrectWarmup` | `nnUNetTrainerHERCWarmup` |
| `nnUNetTrainerSSDMoEFail2CorrectJoint` | `nnUNetTrainerHERCJoint` |
| `ConvNormAct3d` | `ConvNormAct` |
| `ResidualStem3d` | `ResidualStem` |
| `LightweightContextStem3d` | `ContextStem` |
| `LightweightResidualOperator3d` | `ResidualOperator` |

### 3.3 Functions

| Before | After |
|---|---|
| `uncertainty_signal` | `entropy_cue` |
| `ambiguity_signal` | `ambiguity_cue` |
| `spatial_gradient_magnitude` | `probability_gradient_cue` |
| `image_gradient_magnitude` | `image_gradient_cue` |
| `hierarchy_violation` | `hierarchy_violation_cue` |
| `per_region_soft_boundary` | `soft_morphological_band` |
| `probabilities_from_logits` | unchanged |
| `build_failure_targets` | `build_route_targets` |
| `build_region_failure_targets` | `build_region_route_targets` |
| `build_atomic_failure_targets` | `build_atomic_route_targets` |
| `induce_typed_failure` | `inject_typed_error` |
| `induce_region_failure` | `inject_region_error` |
| `induce_atomic_failure` | `inject_atomic_error` |
| `atomic_fg_to_full` | `expand_foreground_to_full` |
| `region_fail2correct_metrics` | `region_correction_metrics` |
| `_infer_nested_region_indices` | `infer_nested_region_indices` |
| `highest_res` | `highest_resolution` |
| `replace_highest` | `replace_highest_resolution` |
| `prepare_static` | `prepare_static_features` |

### 3.4 Configuration fields and tensor keys

| Before | After |
|---|---|
| `expert_residual_bound` | `semantic_residual_bound` |
| `min_step_scale` / `max_step_scale` | `min_step_size` / `max_step_size` |
| `structure_expert` (module attribute) | `hierarchy_expert` |
| `r_semantic_full` / `r_boundary_full` | `r_semantic` / `r_boundary` |
| `r_struct`, `w_struct` | `r_h`, `w_h` |
| `uncertainty` (diagnosis key) | `entropy` |
| `prob_grad` (diagnosis key) | `prob_gradient` |
| `pred_boundary` (boundary key) | `soft_band` |
| `boundary_outward_raw` / `boundary_inward_raw` | `outward_raw` / `inward_raw` |
| `Output['f2c']` | `Output['herc']` |
| per-loss log keys `f2c_<term>` | `herc_<term>` |
| `use_synthetic_failure_warmup` | `use_synthetic_error_warmup` |
| `synthetic_failure_target` / `_type` | `synthetic_error_target` / `_type` |
| `set_fail2correct_enabled` / `_trainable` | `set_herc_enabled` / `set_herc_trainable` |
| `fail2correct_enabled` | `herc_enabled` |
| `_f2c_phase_name` | `_phase_name` |
| state-dict prefix `fail2correct.` | `herc.` |
| env prefix `F2C_` | `HERC_` |

## 4. Consequences

* **Checkpoints.** The correction prefix in `state_dict` changed from
  `fail2correct.` to `herc.`; see Section 8 of [README.md](README.md) for the
  one-line remapping. The frozen backbone keys are unchanged, so baseline SSDMoE
  checkpoints still load.
* **Trainer names.** `-tr` now takes `nnUNetTrainerHERC`,
  `nnUNetTrainerHERCWarmup` or `nnUNetTrainerHERCJoint`. Run directory names
  containing the old trainer name remain valid but will not be reused.
* **Environment variables.** Export `HERC_*` instead of `F2C_*`; the old prefix is
  no longer read.
* **Numbers.** The rename is identifier-only. Every tensor operation, default and
  loss weight is unchanged; see `HERC_VERIFICATION.md`.
