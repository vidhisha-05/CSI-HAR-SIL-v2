# CSI-HAR SIL — Camera-Ready Project

## Purpose

This is the curated camera-ready working project for the CSI-HAR
Selective Invariance study.

The historical development material is retained separately from the active public project and is not part of the public repository.

## Active scientific results

### Original supplied UT-HAR split

Raw CE:
- Accuracy: 99.00%

SIL-v2:
- Accuracy: 96.60%

This result is retained as the original benchmark split result.

### Component-disjoint single split

Raw CE:
- Accuracy: 91.52%
- Macro-F1: 0.8936

SIL-v2:
- Accuracy: 94.28%
- Macro-F1: 0.9197

SIL-v2 improvement:
- Accuracy: +2.76 percentage points
- Macro-F1: +0.0261

Paired permutation:
- p = 0.006850

### Component-disjoint 5-fold

Raw CE:
- Accuracy: 92.66 +/- 0.85%

SIL-v2:
- Accuracy: 93.77 +/- 0.70%

Accuracy improvement:
- +1.11 +/- 0.23 percentage points

Raw CE Macro-F1:
- 0.8901 +/- 0.0066

SIL-v2 Macro-F1:
- 0.9064 +/- 0.0085

Macro-F1 improvement:
- +0.0163 +/- 0.0053

SIL-v2 improves in all five folds.

## Scientific interpretation

The strongest current evidence is based on controlled component-disjoint
evaluation and controlled static-component perturbation.

The work does NOT establish:

- real cross-room generalization;
- causal proof of a room fingerprint;
- complete invariance to all static information;
- recording-independent generalization beyond the constructed
  component-disjoint protocol.

## Project organization

01_DATA
- Raw UT-HAR data

02_CODE
- Current reproducible code

03_RESULTS
- Current experimental results

04_AUDITS
- Provenance and intervention audits

05_CHECKPOINTS
- Four canonical checkpoints

06_FIGURES
- Active figure output only
- Currently intentionally empty

07_FIGURE_DATA
- Verified numerical data used to generate figures

08_PAPER
- Final manuscript package

09_ENVIRONMENT
- Reproducibility information

99_ARCHIVE
- Historical experiments
- Old figures
- Old SIL
- Old baselines
- Old controls
- WIDAR
- Legacy code
- Historical paper workspace

## Rule

Never use files from 99_ARCHIVE for current paper numbers unless explicitly
verified and promoted into the active project.

