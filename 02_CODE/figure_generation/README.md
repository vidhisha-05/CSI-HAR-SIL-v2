
# CSI-HAR Camera-Ready Figure Scripts

## Purpose

These scripts generate a coherent, research-paper-style figure set for the
camera-ready CSI-HAR SIL-v2 manuscript. They are deliberately designed around
the scientific claims rather than generic plotting.

## Main figures

1. `fig01_component_targeted_invariance.py`
   - Method mechanism.
   - Stable component decomposition.
   - Static perturbation.
   - Reconstruction.
   - Shared encoder and consistency objective.

2. `fig02_static_intervention_robustness.py`
   - Static-gain robustness curve.
   - Unseen static transformations.
   - Static-vs-dynamic intervention.
   - Uses the validated paired robustness values.

3. `fig03_strict_component_disjoint_evaluation.py`
   - Temporal-overlap problem.
   - Component-disjoint protocol.
   - Original vs strict performance.
   - Bootstrap CI and paired permutation result.

4. `fig04_component_disjoint_5fold_reproducibility.py`
   - Five paired strict folds.
   - Exact fold-level values from the verified CSV.
   - Aggregate accuracy and Macro-F1 gains.

## Supplementary

5. `figS1_temporal_redundancy_audit.py`
   - Temporal-overlap component-size audit.

## Important design rules

- Do not add fake seeds or fabricated uncertainty.
- Do not describe the component as a proven "room fingerprint".
- Do not claim real cross-room generalization from synthetic perturbations.
- Do not include the generic-consistency control in these paper figures.
- Do not use the obsolete 7.3x robustness ratio.
- Do not reuse old July figures.
- All numbers should remain traceable to the camera-ready results/audits.
- Figures are exported as PDF, SVG, and 600-DPI PNG.
- Matplotlib's default color cycle is intentionally used so the scripts do not
  hard-code a potentially misleading color semantics.

## Running

From the script directory:

    python fig01_component_targeted_invariance.py
    python fig02_static_intervention_robustness.py
    python fig03_strict_component_disjoint_evaluation.py
    python fig04_component_disjoint_5fold_reproducibility.py
    python figS1_temporal_redundancy_audit.py

The scripts expect the camera-ready project at:

    the repository root

They write only into:

    06_FIGURES\main
    06_FIGURES\supplementary

## Deliberately not automated yet

A representation-stability figure should be generated only after extracting
embeddings from the current checkpoints with the exact final encoder. It is
better to leave that figure pending than to silently reuse old embedding
numbers from the pre-SIL-v2 audit.

