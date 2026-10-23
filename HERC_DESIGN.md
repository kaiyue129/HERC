# HERC design rationale

## Central claim

The second stage should **not segment the tumor again**.

It should model the residual error of an already strong SSDMoE predictor:

```text
segment -> diagnose error -> apply typed corrective operator -> re-diagnose
```

## Why this is distinct from SSDMoE

SSDMoE performs adaptive specialization in representation space.
HERC performs specialization in error space.

```text
SSDMoE experts:      what semantic transformation should this token receive?
HERC experts: which typed error does the final predictor make at this location?
```

## Connection to FineSAM++

FineSAM++ motivates:

- spatially selective correction
- residual rather than full re-prediction
- expert diversity induced by different error conditions
- progressive training after a stable coarse model exists

HERC changes the implementation for 3D BraTS:

- binary error gate -> continuous soft severity
- generic residual experts -> typed error operators
- random degradation -> typed error-space augmentation
- 2D local correction -> cross-scale 3D correction

## Connection to CausalBridgeNet

CausalBridgeNet motivates:

- closed-loop prediction-error feedback
- directional false-positive/false-negative correction
- additive correction in prediction/logit space

HERC extends this to:

- spatial error routing instead of a volume-global route
- explicit WT/TC/ET hierarchy repair
- full-resolution inward/outward boundary actions
- repeated diagnose-correct cycles
- safety and monotonicity constraints

## Region-based BraTS error decomposition

For sigmoid region probabilities `p=[WT,TC,ET]` and targets `y`:

### FN severity

`max_c y_c (1-p_c)`

### FP severity

`max_c (1-y_c) p_c`

### Hierarchy violation

`ReLU(p_TC-p_WT) + ReLU(p_ET-p_TC)`

### Boundary error

prediction error restricted to GT region boundary bands.

These four scores form a soft routing target rather than mutually exclusive hard labels.

## Safety philosophy

A strong baseline is mostly correct. Therefore the corrector must learn both:

1. how to repair errors;
2. when not to change a voxel.

Safety mechanisms:

- low initial severity bias
- near-zero corrective residual initialization
- bounded logit updates
- learned step scale
- monotonic step penalty
- corrected-vs-initial safety penalty
- adaptive stop at inference

## Why the Boundary Expert is full resolution

The semantic experts work at the penultimate decoder stage for efficiency.
Upsampling their correction cannot precisely recover sub-voxel/surface geometry at the 128^3 output grid.

The boundary expert therefore uses the final decoder feature and operates only in a gated narrow band.
Its capacity is intentionally tiny to keep the explanation as **precision correction**, not increased model size.
