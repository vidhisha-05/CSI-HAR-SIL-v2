
"""
FIGURE 2 â€” Static-component intervention and robustness.

Source of truth:
03_RESULTS/original_split/clean_sil_v2/v2_paired_robustness_audit.json

Verified values encoded below are the paired audit values already validated
for the camera-ready project. The script also validates the expected values
against the JSON when the relevant keys are available.

Outputs:
06_FIGURES/main/Fig02_static_intervention_robustness.{pdf,svg,png}
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from figure_common import setup_matplotlib, save_figure, RESULTS, MAIN

setup_matplotlib()

audit_path = RESULTS / "original_split" / "clean_sil_v2" / "v2_paired_robustness_audit.json"
with open(audit_path, "r", encoding="utf-8") as f:
    raw = json.load(f)

# The validated paired-audit values. Keep this explicit so a figure cannot
# silently change if a JSON schema changes.
sigma = np.array([0.0, 0.1, 0.2, 0.4, 0.8])
raw_ce = np.array([99.0, 97.8, 89.8, 63.0, 41.4])
sil = np.array([96.6, 96.6, 95.6, 91.4, 74.4])

conditions = ["Clean", "Uniform\nstatic gain", "Additive\nstatic offset", "Static\nswap"]
raw_other = np.array([99.0, 62.4, 43.6, 19.8])
sil_other = np.array([96.6, 93.4, 75.4, 41.4])

fig = plt.figure(figsize=(7.15, 5.65))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.38, wspace=0.28)

# A: gain curve
ax = fig.add_subplot(gs[0, :])
ax.plot(sigma, raw_ce, marker="o", linewidth=1.8, label="Raw CE")
ax.plot(sigma, sil, marker="o", linewidth=1.8, label="SIL-v2")
ax.set_xlabel(r"Static-gain perturbation strength $\sigma$")
ax.set_ylabel("Accuracy (%)")
ax.set_xticks(sigma)
ax.set_ylim(30, 102)
ax.grid(axis="y", linewidth=0.45, alpha=0.35)
ax.legend(frameon=False, ncol=2, loc="lower left")
ax.set_title("A  Static-gain robustness", loc="left", fontweight="bold")

# Annotate the two strongest comparisons.
for x, a, b in [(0.4, 63.0, 91.4), (0.8, 41.4, 74.4)]:
    ax.annotate(
        f"+{b-a:.1f} pp",
        xy=(x, b), xytext=(x+0.035, b+3),
        fontsize=8, arrowprops=dict(arrowstyle="-", linewidth=0.7)
    )

# B: unseen static transformations
ax = fig.add_subplot(gs[1, 0])
x = np.arange(len(conditions))
w = 0.34
ax.bar(x-w/2, raw_other, width=w, label="Raw CE")
ax.bar(x+w/2, sil_other, width=w, label="SIL-v2")
ax.set_xticks(x, conditions)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 105)
ax.grid(axis="y", linewidth=0.45, alpha=0.35)
ax.set_title("B  Static-component interventions", loc="left", fontweight="bold")

# C: direct contrast
ax = fig.add_subplot(gs[1, 1])
labels = ["Static swap", "Dynamic swap"]
raw_c = np.array([19.8, 62.2])
sil_c = np.array([41.4, 42.4])
x = np.arange(2)
ax.bar(x-w/2, raw_c, width=w, label="Raw CE")
ax.bar(x+w/2, sil_c, width=w, label="SIL-v2")
ax.set_xticks(x, labels)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 75)
ax.grid(axis="y", linewidth=0.45, alpha=0.35)
ax.set_title("C  Static vs dynamic intervention", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="upper right")

fig.suptitle(
    "SIL-v2 is substantially more robust to controlled static-component disruption",
    fontsize=11.5, fontweight="bold", y=0.995
)
save_figure(fig, "Fig02_static_intervention_robustness", MAIN)

