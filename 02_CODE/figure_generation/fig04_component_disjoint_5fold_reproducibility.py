
"""
FIGURE 4 â€” Five-fold component-disjoint reproducibility.

The fold-level values are read from:
07_FIGURE_DATA/component_disjoint_5fold_exact.csv

This is intentionally a paired dumbbell/slope plot, not a basic grouped bar
chart. It shows the direction and magnitude of the improvement in every fold.

Outputs:
06_FIGURES/main/Fig04_component_disjoint_5fold_reproducibility.{pdf,svg,png}
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from figure_common import setup_matplotlib, save_figure, FIGURE_DATA, MAIN

setup_matplotlib()

csv_path = FIGURE_DATA / "component_disjoint_5fold_exact.csv"

folds, raw, sil = [], [], []
with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        folds.append(int(row["Fold"]))
        raw.append(float(row["Raw CE Accuracy"]))
        sil.append(float(row["SIL-v2 Accuracy"]))

folds = np.asarray(folds)
raw = np.asarray(raw)
sil = np.asarray(sil)
delta = sil - raw

assert len(folds) == 5
assert np.all(delta > 0), "Expected SIL-v2 to win every fold."

fig, ax = plt.subplots(figsize=(6.9, 4.25))

y = np.arange(len(folds))[::-1]

for yi, r, s, d, fold in zip(y, raw, sil, delta, folds):
    ax.plot([r, s], [yi, yi], linewidth=2.0)
    ax.scatter(r, yi, s=42, label="Raw CE" if fold == 1 else None, zorder=3)
    ax.scatter(s, yi, s=42, label="SIL-v2" if fold == 1 else None, zorder=3)
    ax.text(s + 0.12, yi, f"+{d:.2f} pp", va="center", fontsize=8)

ax.set_yticks(y, [f"Fold {f}" for f in folds])
ax.set_xlabel("Component-disjoint test accuracy (%)")
ax.set_xlim(90.5, 95.8)
ax.grid(axis="x", linewidth=0.45, alpha=0.35)
ax.legend(frameon=False, ncol=2, loc="lower right")
ax.set_title("SIL-v2 improves accuracy in all five strict component-disjoint folds",
             loc="left", fontweight="bold")

ax.text(
    0.01, -0.19,
    "Aggregate: Raw CE 92.66 Â± 0.85%  |  SIL-v2 93.77 Â± 0.70%\n"
    "Accuracy gain: +1.11 Â± 0.23 pp  |  Macro-F1 gain: +0.0163 Â± 0.0053",
    transform=ax.transAxes, fontsize=8.2, va="top"
)

save_figure(fig, "Fig04_component_disjoint_5fold_reproducibility", MAIN)

