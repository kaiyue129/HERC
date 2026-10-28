# HERC design rationale

Why the correction cell is built the way it is. See [README.md](README.md) for the
manuscript-to-code map, the reported settings and the training procedure; the
section numbers below refer to the manuscript.

## Central claim

Localising a residual error does not specify how to repair it. A missed region
needs added foreground, a spurious region needs it removed, a nesting violation
needs a joint adjustment of two region heads, and a displaced surface needs a local
full-resolution move. The module therefore separates *where* to intervene from
*which action* is permitted there (Section 3.2.1):

```text
diagnose the current state -> allocate the permissible actions
-> apply a bounded update -> diagnose the resulting state again
```

## Position relative to prior refinement

FineSAM++ already combines confidence-driven spatial soft routing with residual
experts, and CausalBridgeNet already feeds structured prediction errors back through
iterative correction while keeping a frozen foundation model (Section 2.2). Neither
soft routing nor prediction-driven feedback alone is specific to HERC. The
contribution is the coordination of four explicitly defined correction roles inside
one shared cell, softly combined in a bounded update that is reassessed at the next
iteration.

## Region-based error decomposition

The route targets of Section 3.2.5 are built from the current region probabilities
`p = [WT, TC, ET]` and the reference `y`:

- false-negative evidence: `max_c y_c (1 - p_c)`
- false-positive evidence: `max_c (1 - y_c) p_c`
- hierarchy violation: `ReLU(p_TC - p_WT) + ReLU(p_ET - p_TC)`
- boundary evidence: the prediction error restricted to the reference boundary band

These four scores form a soft routing target, not mutually exclusive labels, so
several routes may contribute at the same voxel.

## Where the constraints live

Only the false-negative and false-positive experts are sign-restricted
(`b_s σ(·)` and `−b_s σ(·)`). Their weighted mixture may also include hierarchy and
boundary contributions and is not constrained to a single sign (Section 3.2.3). The
severity map gates the coarse semantic branch; the boundary branch has its own
full-resolution support and is not multiplied by it (Section 3.2.1).

## Why the boundary branch is full resolution

The semantic experts work at the penultimate decoder stage, which is where the
region heads and the diagnostic cues are cheap to evaluate. Upsampling their
residual cannot place an edge on the output grid, so the boundary route predicts at
128³ from the full-resolution feature, the probabilities, the upsampled boundary
weight and the image gradient, and acts only inside a soft morphological band. Its
hidden width is deliberately small: the branch is a precise, low-capacity correction
rather than an increase in model size (Section 3.2.4).

## Preservation

A strong baseline is mostly correct, so the corrector has to learn both how to
repair an error and when to leave a voxel alone. The released configuration uses two
terms for this: the monotonicity term, which penalises an increase of the
segmentation loss between steps, and the safety term, which anchors the corrected
prediction to the initial state (Section 3.2.5). Together with the trust-region
bound on the fused residual and the learned step size, they keep the update small
where the diagnosis is weak.
