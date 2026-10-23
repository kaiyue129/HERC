# Verification status

## 1. Checks that passed in this environment

### 1.1 Before the rename — functional evidence of the pre-rename package

These were run when the package was built, on the pre-rename identifiers. They are
still the functional evidence for the module; the rename itself is covered by
Section 2.

- All Python files under `umamba/nnunetv2` compile: 178 files, 0 syntax errors.
- Region-mode core forward test passed.
- Two-step shared-weight correction passed.
- FN residual non-negativity and FP residual non-positivity checks passed.
- Hierarchy-violation detection passed.
- Typed-error augmentation shape/semantics test passed.
- Composite HERC loss with an nnU-Net-like deep-supervision wrapper passed forward
  and backward.
- Adaptive inference path with top-k severity stopping logic passed structural
  execution.
- Analytical static audit script generated `herc_static_audit.json`.
- Baseline-to-HERC state-dict compatibility: new keys are restricted to `herc.*`.

### 1.2 After the rename — static evidence that the change is identifier-only

`legacy/qa/verify_rename.py` performs four checks and all of them pass:

```bash
python legacy/qa/verify_rename.py --old /path/to/pre-rename/copy
# [naming]   OK    no pre-rename identifier outside NAMING.md / README.md / legacy/
# [imports]  OK    188 files parsed, every intra-project import resolves and
#                  every imported name is defined in its target module
# [defaults] OK    every dataclass and __init__ default value unchanged
# [layers]   OK    identical multiset of layer-constructor calls, so the
#                  17,678-parameter budget and the channel arithmetic are intact
```

Additionally, `py_compile` succeeds on all 188 Python files of the renamed tree.

## 2. Deliberate behavioural differences of the rename

The rename was designed to be identifier-only. Three points are worth stating
explicitly, because a diff shows them:

| Change | Why it is safe |
|---|---|
| the configuration keyword argument and the state-dict prefix were renamed (`herc_config`, `herc.`) | pure renaming; a baseline SSDMoE checkpoint still loads, an old HERC checkpoint needs the one-line remap in README Section 8 |
| the boolean environment flags are read with the original inline `os.environ.get(...).lower() in ('1', 'true', 'yes')` expression | no helper function was introduced, so the parsed value is identical |
| the boundary expert computes the soft band in one branch instead of two | both branches previously assigned the same tensor (`pred_boundary`) to the band input; the region branch uses the region probabilities, the atomic branch uses `1 - p_background`, exactly as before |

## 3. Analytical results (128^3, B=1, region mode, 64/32 decoder taps)

- HERC parameters: **17,678 (0.0177 M)**
- convolution-dominated complexity:
  - static once: 1.074 GMAC
  - dynamic per correction step: 5.870 GMAC
  - T=2: **12.814 GMAC**
  - strict 2-op convention: 25.627 GFLOPs
- scaled unique saved spatial activations for the T=2 AMP graph: about **0.69 GiB**
- conservative correction-only backward overhead budget: roughly **1.0–1.5 GiB**

See `HERC_CODE_AUDIT.md` for the methodology and the tensor-by-tensor details.

## 4. Not possible in this sandbox

A complete SSDMoE + Mamba-2 CUDA forward cannot be executed here: the environment
does not include the runtime dependencies (`mamba_ssm`,
`dynamic-network-architectures`, `batchgenerators`, `acvl-utils`) and does not
expose the target CUDA runtime. `torch` itself is not installed, so the two unit
tests were not re-executed after the rename; their correctness under the new names
rests on the static checks of Section 1.2.

Consequently the following still have to be measured in the original SSDMoE
environment:

- CUDA peak allocated/reserved memory,
- wall-clock training step,
- sliding-window inference latency,
- exact supported-operator GFLOPs under the baseline profiler.

The project includes benchmark code for this purpose.

## 5. Recommended post-rename smoke test (on the GPU machine)

```bash
pip install -e umamba            # make this nnunetv2 shadow site-packages
python -m nnunetv2.tests.herc.test_herc_core
python -m nnunetv2.tests.herc.test_herc_loss
python -m nnunetv2.run.run_herc_static_audit --patch 128 --steps 2
```
