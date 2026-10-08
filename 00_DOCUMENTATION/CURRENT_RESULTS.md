# CSI-HAR SIL Camera-Ready — Current Results

## Original supplied UT-HAR split

Raw CE:
- Test accuracy: 99.00%

SIL-v2:
- Test accuracy: 96.60%

Difference:
- SIL-v2: -2.40 percentage points

## Component-disjoint single split

Raw CE:
- Test accuracy: 91.52%
- Macro-F1: 0.8936

SIL-v2:
- Test accuracy: 94.28%
- Macro-F1: 0.9197

Difference:
- Accuracy: +2.76 percentage points
- Macro-F1: +0.0261

Paired permutation:
- Two-sided p = 0.006850

## Component-disjoint 5-fold

Fold 1:
- Raw CE: 93.8693%
- SIL-v2: 94.8744%
- Delta: +1.0050 pp

Fold 2:
- Raw CE: 92.9577%
- SIL-v2: 93.7626%
- Delta: +0.8048 pp

Fold 3:
- Raw CE: 92.5628%
- SIL-v2: 93.6683%
- Delta: +1.1055 pp

Fold 4:
- Raw CE: 91.5323%
- SIL-v2: 92.9435%
- Delta: +1.4113 pp

Fold 5:
- Raw CE: 92.3771%
- SIL-v2: 93.5807%
- Delta: +1.2036 pp

Aggregate previously reported:
- Raw CE accuracy: 92.66 +/- 0.85%
- SIL-v2 accuracy: 93.77 +/- 0.70%
- Accuracy improvement: +1.11 +/- 0.23 pp

- Raw CE Macro-F1: 0.8901 +/- 0.0066
- SIL-v2 Macro-F1: 0.9064 +/- 0.0085
- Macro-F1 improvement: +0.0163 +/- 0.0053

## Interpretation

The component-disjoint evaluation is the stronger generalization protocol because
exact 150-row CSI blocks were separated across the newly constructed partitions.

The current evidence supports:
- improved performance under the component-disjoint protocol,
- robustness to controlled static-component perturbations,
- and consistent SIL-v2 improvement across all five component-disjoint folds.

It does NOT establish:
- real cross-room generalization,
- causal proof of a room fingerprint,
- recording-independent generalization beyond the constructed component-disjoint protocol,
- or complete invariance to static information.
