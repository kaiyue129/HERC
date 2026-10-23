# HERC — defect audit and fixes applied

Date of audit: after the package was delivered.
Scope: **code defects only**. Nothing in the method design, configuration defaults,
loss weighting, metric definitions or training protocol was changed.

Verdict of the audit: **the code structure is complete and functionally correct.**
A 17-check functional sweep (including the atomic/softmax fallback path, which had
never been exercised before) passes in full. Exactly three concrete defects were
found, all of them non-algorithmic. They are fixed below.

---

## 1. Fixes applied

### FIX-1 / FIX-2 — shipped tests could not run at all

Files:

- `umamba/nnunetv2/tests/herc/test_herc_core.py`
- `umamba/nnunetv2/tests/herc/test_herc_loss.py`

The tests computed their project root with `parents[4]`, which resolves to
`HERC/` — a directory that does **not** contain an `nnunetv2`
package (the package lives in `HERC/umamba/nnunetv2/`).
The correct index is `parents[3]`.

Observed before the fix:

```text
ModuleNotFoundError: No module named 'nnunetv2.nets'
```

This was misleading, because a stock `nnunetv2` installed in site-packages was
picked up instead, and that copy has no `.nets` submodule.

```diff
-ROOT = Path(__file__).resolve().parents[4]
+ROOT = Path(__file__).resolve().parents[3]
```

Both tests now run as shipped, from any working directory.

### FIX-3 — `_infer_nested_region_indices` was defined three times

File: `umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainerHERC.py`

The helper was pasted three times (originally at lines 24, 305 and 340). The three
copies were verified **byte-identical** (same MD5, 781 bytes each) before removal,
so deleting the two redundant copies is provably behaviour-preserving.

The two duplicates that sat between the trainer classes were removed. One
definition remains, at module top (line 24). The file went from 380 to 334 lines.
All three trainer classes and their inheritance are unchanged:

```text
nnUNetTrainerHERC          (base nnUNetTrainerSSDMoEUNet)  9 methods
nnUNetTrainerHERCWarmup    (base ...HERC)          1 method
nnUNetTrainerHERCJoint     (base ...HERC)          1 method
```

---

## 2. Post-fix verification

| Check | Result |
|---|---|
| `compileall` over `umamba/nnunetv2` | pass |
| `compileall` over `evaluation` | pass |
| `test_herc_core.py` (as shipped, neutral cwd) | pass |
| `test_herc_loss.py` (as shipped, neutral cwd) | pass, loss `1.0638819932937622` (identical to pre-fix) |
| duplicate top-level definition scan | none remaining |
| 17-check functional sweep (region + atomic + gradients + init + stopping) | all pass |
| real `128^3` forward shape/finiteness smoke test | pass, 4.5 s, peak RSS 2.04 GiB |
| import-chain validation (with `mamba_ssm` stubbed) | pass |

Independent cross-checks obtained during the audit:

- HERC parameter count **17,678** — reproduces `herc_static_audit.json`
  exactly.
- Decoder taps `(64, 32)` for a base-width-32 plan — reproduces
  `HERC_CODE_AUDIT.md` exactly.
- `_infer_nested_region_indices` returns `(0,1,2)` for the BraTS region order,
  `(1,2,0)` for a permuted region order, and raises on a non-nested chain.

---

## 3. Deliberately NOT changed

These were examined and left alone. They are design/protocol decisions or harmless
residue, not defects. Changing them silently would alter experimental semantics.

| Item | Why left alone |
|---|---|
| Training uses `train_steps=2` while validation/inference uses `max_inference_steps=3` with adaptive stopping | Real protocol inconsistency, but it is a **configuration decision**. Fix it by exporting `HERC_MAX_INFERENCE_STEPS=2` and `HERC_ADAPTIVE_STOP=0` for the main accuracy experiment. Do not change the defaults, or existing runs become non-reproducible. |
| `harm_rate` uses all correctly-predicted voxels (including background) as denominator | The code correctly implements the stated definition; the definition is diluted by background. This is a metric-design matter (add ROI-restricted variants) rather than a bug. |
| `evaluation/herc_metrics.py` assumes WT/TC/ET = 0/1/2 | Latent only. The trainer infers the order from `label_manager.foreground_regions`, and for the BraTS dataset the inferred order is always 0/1/2, so the two agree. |
| `inject_region_error` modes `fn` and `boundary`(inward) produce the identical perturbation; `fp` and `boundary`(outward) differ only in kernel radius (5 vs 3) | An **error-space augmentation design** issue that affects the planned typed-vs-random ablation, not a code defect. Fixing it means redesigning the augmentation, which is an experimental decision. |
| `region_aux_seg_loss` / `atomic_aux_seg_loss` / `aux_seg_loss` are defined but never called | Dead public helpers. Harmless; removing them could break an external import. |
| `load_pretrained_weights.py:21` assigns `skip_strings_in_pretrained = None` and reassigns it at line 31 | Dead assignment, no behavioural effect. |
| GSAF in `SSDMoEUNet.py` builds a zero-channel conv when a stage width is < 32 (`dim // reduction` with `reduction=32`) | Real constraint, but it belongs to the **baseline** file the extension must not modify, and it is never triggered by the BraTS plan (widths >= 32). |
| `PROJECT_TREE.txt` lists `__pycache__` entries; `legacy/FAIL2CORRECT_pre_rename.patch` is now stale | Documentation artifacts. See the note below. |

---

## 4. Note on the bundled patch and tree listing

`legacy/FAIL2CORRECT_pre_rename.patch` and `PROJECT_TREE.txt` were generated before these fixes and
therefore still contain the pre-fix versions of the two test files and the
duplicated helper.

**Use the `HERC/` directory as the deliverable.** If the patch
is applied to a checkout instead, apply the three changes in section 1 on top of it,
or the test-path bug will be reintroduced.

---

## 5. Environment facts relevant to running the experiments

These are not code defects, but they will block experiments if unaddressed.

1. **`mamba_ssm` is absent and is not declared in `umamba/setup.py`.** `SSDMoEUNet.py`
   imports it at module top level (`from mamba_ssm import Mamba2`). It is CUDA-only,
   so the SSDMoE forward cannot run on a CPU-only machine.
2. **A stock `nnunetv2 2.8.0` is installed in site-packages, while this project is
   `nnunetv2 2.1.1`.** The installed copy contains no `nets/SSDMoEUNet.py` and no
   `nets/herc/`. Unless the project is installed so that it shadows
   site-packages (for example `pip install -e HERC/umamba`, ideally
   in a dedicated environment), `nnUNetv2_train -tr nnUNetTrainerHERC...`
   will not resolve to this code.

   Self-check:

   ```bash
   python -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))"
   # must print ...\HERC\umamba\nnunetv2
   ```

3. **`wandb` is a hard import in `nnUNetTrainer.py`** (pre-existing in the SSDMoE
   repository, not a HERC change). It must be installed even when unused.
