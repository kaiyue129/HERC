# SSDMoE-HERC — Experiment Implementation & Code Audit

## 1. Audit scope

Reviewed execution chain:

`HERCUNet.py -> diagnosis.py -> experts.py -> cell.py -> herc_loss.py -> nnUNetTrainerHERC.py`

The audit is aligned with the uploaded SSDMoE-UNet repository and its nnU-Net region-based BraTS pipeline.

## 2. Real BraTS data contract

For the repository's BraTS region training:

- image patch: `[B, 4, 128, 128, 128]`
- batch size: `1`
- network output: three independent region logits, not a 4-class softmax
- head semantics: nested WT / TC / ET regions
- highest-resolution target: `[B, 3, 128, 128, 128]`
- deep-supervision target: list of region tensors at successively lower spatial scales
- inference nonlinearity: sigmoid per region head

The trainer now infers the WT/TC/ET head indices from the actual nnU-Net region definitions by set inclusion rather than assuming channel order.

## 3. Baseline feature taps

`UNetResDecoder` records decoder channels dynamically and returns:

- `decoder_coarse = decoder_features[-2]`
- `decoder_full   = decoder_features[-1]`

With the standard nnU-Net base width of 32, the relevant 128^3 path is:

- `decoder_coarse`: `[1, 64, 64, 64, 64]`
- `decoder_full`: `[1, 32, 128, 128, 128]`
- baseline region logits: `[1, 3, 128, 128, 128]`

The code does not hard-code those channels; `HERCUNet` reads them from `decoder.stage_output_channels`.

## 4. Tensor flow per correction iteration

### 4.1 Static tensors cached once across all correction iterations

| Tensor | Shape | FP16 storage |
|---|---:|---:|
| MRI | `[1,4,128,128,128]` | 16 MiB |
| coarse decoder tap | `[1,64,64,64,64]` | 32 MiB |
| full decoder tap | `[1,32,128,128,128]` | 128 MiB |
| projected coarse feature | `[1,32,64,64,64]` | 16 MiB |
| projected full feature | `[1,8,128,128,128]` | 32 MiB |
| MRI gradient cue | `[1,1,128,128,128]` | 4 MiB |

The projected features and MRI gradient are computed once and reused for `t=0,1,...`.

### 4.2 HERCDiagnosis

At 64^3:

- current probabilities: `[1,3,64,64,64]`
- uncertainty: `[1,1,64,64,64]`
- ambiguity: `[1,1,64,64,64]`
- probability gradient: `[1,1,64,64,64]`
- hierarchy violation: `[1,1,64,64,64]`
- concatenated diagnosis input: `[1,39,64,64,64]`
- diagnosis context: `[1,32,64,64,64]`
- severity: `[1,1,64,64,64]`
- route logits/weights: `[1,4,64,64,64]`
- step scale logit: `[1,1]`

Route order in region mode:

1. FN recovery
2. FP suppression
3. hierarchy consistency
4. boundary repair

### 4.3 Typed semantic experts

Input to each semantic expert:

`[diagnosis context (32), current region probabilities (3)] -> [1,35,64,64,64]`

Outputs:

- FN expert: `[1,3,64,64,64]`, structurally non-negative
- FP expert: `[1,3,64,64,64]`, structurally non-positive
- hierarchy expert: `[1,3,64,64,64]`, only acts on `TC>WT` or `ET>TC` probability violations

The semantic mixture remains `[1,3,64,64,64]` and is trilinearly upsampled to `[1,3,128,128,128]`.

### 4.4 BoundaryCorrectionExpert

At 128^3:

- current region probabilities: `[1,3,128,128,128]`
- per-region soft boundary: `[1,3,128,128,128]`
- boundary route: `[1,1,128,128,128]`
- per-region boundary gate: `[1,3,128,128,128]`
- projected decoder feature: `[1,8,128,128,128]`
- MRI gradient: `[1,1,128,128,128]`
- boundary branch input: `[1,16,128,128,128]`
- boundary hidden representation: `[1,8,128,128,128]`
- outward + inward raw actions: `[1,6,128,128,128]`
- signed boundary residual: `[1,3,128,128,128]`

The boundary branch is activation-checkpointed during training.

### 4.5 Logit update

For each iteration:

`DeltaZ_t = semantic_residual + boundary_residual`

then a bounded trust-region transform is applied and the per-volume adaptive step scale `eta_t` is used:

`Z_(t+1) = Z_t + eta_t * DeltaZ_t`

The same HERC cell parameters are shared across all iterations.

## 5. Corrections made during this audit

### 5.1 Diagnosis compute was too expensive

Previous version used two dense 3x3x3 convolutions at 64^3. Those two layers alone dominated the new method's compute.

They were replaced by a lightweight context stem:

`1x1 projection -> depthwise 3x3x3 -> pointwise 1x1x1 -> depthwise 3x3x3`

The full-resolution boundary stem was changed in the same way.

### 5.2 Added activation checkpointing to low-resolution correction blocks

The diagnosis stem and typed semantic expert operators now support non-reentrant checkpointing. The full-resolution boundary operator was already checkpointed.

### 5.3 Static features are cached across correction iterations

The decoder projections and image-gradient cue are iteration-invariant. They are now computed once before the correction loop, avoiding duplicate full-resolution work for T=2/T=3.

### 5.4 Adaptive stopping no longer averages over the full brain volume

A whole-volume mean severity is unsafe for sparse lesions because background dominates the statistic.

The code now evaluates the mean of the top severity fraction (`stopping_topk_fraction`, default 2%) before deciding whether to continue correction.

For the main accuracy experiment, fixed `T=2` is still recommended; adaptive stopping should be a later efficiency ablation after validation-set calibration.

### 5.5 WT/TC/ET head order is no longer hard-coded

The trainer now derives the three nested region heads from `label_manager.foreground_regions` and validates:

`ET subset TC subset WT`.

### 5.6 Corrected high-resolution loss is no longer diluted by frozen deep-supervision outputs

HERC only changes the highest-resolution prediction. Therefore the main correction loss now uses the exact nnU-Net primitive Dice+BCE/CE on the highest-resolution output.

Deep supervision is retained for the baseline branch during joint fine-tuning, where it is actually useful.

### 5.7 Synthetic benchmark now supports region targets

`run_efficiency_benchmark.py` previously generated an atomic `[B,1,...]` target even when the trainer used regions. It now creates nested region targets compatible with WT/TC/ET training.

## 6. Parameter count

For region mode with decoder taps 64/32:

| Component | Parameters |
|---|---:|
| Diagnosis | 7,078 |
| FN expert | 2,979 |
| FP expert | 2,979 |
| hierarchy expert | 3,004 |
| boundary expert | 1,638 |
| **HERC total** | **17,678** |

So the correction system adds only **0.0177 M parameters**.

Using the reported SSDMoE baseline parameter count of 157.21 M, the rounded total is approximately **157.23 M**.

## 7. Analytical convolution complexity

The following numbers count convolution/linear multiply-accumulate operations (GMAC). Interpolation, pooling, sigmoid, elementwise routing and morphology operations are omitted because they are relatively small and profiler-dependent.

Static feature projections, once per forward:

- **1.074 GMAC**

Dynamic correction cell per iteration:

- **5.870 GMAC / step**

Therefore:

| Correction setting | Conv-dominated GMAC |
|---|---:|
| T=1 | 6.944 |
| **T=2 (main)** | **12.814** |
| T=3 | 18.684 |

Under a strict convention where multiply and add count separately, T=2 is about **25.63 GFLOPs**.

The SSDMoE manuscript reports 809.82 GFLOPs using its own supported-operator profiler. To make a publication-grade comparison, baseline and HERC must be rerun with exactly the same profiler. If that profiler treats a MAC as one operation, the T=2 convolution increment is about **1.58%** before the omitted lightweight elementwise operations.

## 8. Activation and peak-memory estimate

Important standalone FP16 tensor sizes:

| Tensor | FP16 storage |
|---|---:|
| decoder_full `[1,32,128^3]` | 128 MiB |
| boundary concat `[1,16,128^3]` | 64 MiB |
| boundary hidden `[1,8,128^3]` | 32 MiB |
| boundary raw `[1,6,128^3]` | 24 MiB |
| decoder_coarse `[1,64,64^3]` | 32 MiB |
| diagnosis concat `[1,39,64^3]` | 19.5 MiB |
| diagnosis context `[1,32,64^3]` | 16 MiB |

A saved-tensor graph audit was run on a spatially reduced but channel-identical T=2 model, then spatial tensors were scaled to the real 64^3/128^3 shapes and FP16 storage.

- unique saved spatial activation estimate: ~**0.69 GiB**
- conservative saved-reference estimate: ~**0.89 GiB**
- allowing transient workspace/live tensors, correction-only backward overhead should be budgeted at roughly **1.0-1.5 GiB** beyond the frozen baseline forward.

These are analytical estimates, not CUDA measurements.

### Phase-level peak-memory expectations

The SSDMoE manuscript reports:

- original full-training peak: **11.69 GiB**
- inference peak: **3.54 GiB**

Expected HERC ranges on the same style of hardware/protocol:

| Phase | Estimated peak |
|---|---:|
| frozen SSDMoE + HERC training | **~4.5-6.0 GiB** |
| joint: seg head only | ~5-6.5 GiB |
| joint: final decoder stage | **~7-10 GiB** |
| joint: high-res GSAF additionally | ~9-12 GiB |
| optional full SSD-MoE unfreeze | **~12.5-14.5+ GiB** |

The last row is intentionally not the default. On a 16 GiB RTX 4080 SUPER it has little safety margin once allocator fragmentation and CUDA workspaces are included.

## 9. Recommended experiment order

### Phase A — typed-error warm-up

- load the SSDMoE checkpoint
- freeze the entire SSDMoE network
- T=1
- train typed correction operators on synthetic FN/FP/hierarchy/boundary errors

### Phase B — natural error correction

- still freeze SSDMoE
- T=2
- train on real SSDMoE residual errors
- this is the primary hypothesis test

### Phase C — boundary / iteration ablation

Run in order:

1. semantic correctors only
2. + full-resolution boundary expert
3. + second correction iteration
4. + adaptive stopping (efficiency experiment, not primary accuracy setting)

### Phase D — conservative joint fine-tuning

1. unfreeze segmentation heads
2. then final decoder stage if useful
3. optionally high-resolution GSAF
4. keep SSD-MoE frozen in the main model

## 10. Remaining validation that must be done on the real GPU environment

The current sandbox cannot instantiate the complete SSDMoE model because CUDA `mamba_ssm`, `dynamic-network-architectures`, `batchgenerators`, and `acvl-utils` are absent.

Before long training, run on the actual SSDMoE environment:

1. one real batch forward with shape logging
2. one forward+backward with AMP
3. `torch.cuda.max_memory_allocated()` and `max_memory_reserved()`
4. baseline and proposed efficiency benchmark using identical plans/checkpoints
5. fixed T=1, T=2, T=3 latency measurements
6. verify adaptive-stop severity calibration on the validation split

The project includes `run_herc_static_audit.py` for analytical reporting and the corrected `run_efficiency_benchmark.py` for real GPU measurements.
