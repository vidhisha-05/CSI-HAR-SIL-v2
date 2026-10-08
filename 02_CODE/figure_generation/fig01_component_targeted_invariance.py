
"""
FIGURE 1 â€” Component-targeted invariance mechanism.

Purpose:
Camera-ready conceptual/method figure. This is deliberately a scientific
schematic, not a generic flowchart. It shows the exact decomposition used
by SIL-v2 and the training objective.

Outputs:
06_FIGURES/main/Fig01_component_targeted_invariance.{pdf,svg,png}
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figure_common import setup_matplotlib, save_figure, MAIN

setup_matplotlib()

fig = plt.figure(figsize=(7.2, 4.25))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

def box(x, y, w, h, title, body, fontsize=8.0):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.0,
        fill=False,
    )
    ax.add_patch(patch)
    ax.text(x+w/2, y+h*0.70, title, ha="center", va="center",
            fontweight="bold", fontsize=9)
    ax.text(x+w/2, y+h*0.36, body, ha="center", va="center",
            fontsize=fontsize, linespacing=1.35)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=12,
        linewidth=1.0
    ))

# Input
box(0.035, 0.62, 0.17, 0.22,
    "Input CSI",
    r"$X(t,k)$" "\n" "250 Ã— 90 CSI window")

# Decomposition
box(0.27, 0.72, 0.20, 0.17,
    "Stable component",
    r"$S(k)=\mathrm{mean}_t\,X(t,k)$")

box(0.27, 0.47, 0.20, 0.17,
    "Temporal residual",
    r"$D(t,k)=X(t,k)-S(k)$")

arrow(0.205, 0.73, 0.27, 0.79)
arrow(0.205, 0.68, 0.27, 0.55)

# Perturbation
box(0.53, 0.72, 0.20, 0.17,
    "Targeted perturbation",
    r"$\widetilde S(k)=(1+g_k)S(k)$" "\n"
    r"$g_k\sim\mathcal N(0,\sigma^2)$")

box(0.53, 0.47, 0.20, 0.17,
    "Reconstruction",
    r"$\widetilde X(t,k)=\widetilde S(k)+D(t,k)$")

arrow(0.47, 0.80, 0.53, 0.80)
arrow(0.47, 0.55, 0.53, 0.55)
arrow(0.63, 0.72, 0.63, 0.64)

# Encoders
box(0.785, 0.72, 0.17, 0.17,
    "Shared encoder",
    r"$z=f_\theta(X)$")

box(0.785, 0.47, 0.17, 0.17,
    "Shared encoder",
    r"$\widetilde z=f_\theta(\widetilde X)$")

arrow(0.73, 0.80, 0.785, 0.80)
arrow(0.73, 0.55, 0.785, 0.55)

# Objective band
obj = FancyBboxPatch(
    (0.20, 0.10), 0.60, 0.23,
    boxstyle="round,pad=0.015,rounding_size=0.012",
    linewidth=1.1, fill=False
)
ax.add_patch(obj)
ax.text(0.50, 0.285, "Training objective", ha="center", va="center",
        fontweight="bold", fontsize=9)
ax.text(
    0.50, 0.185,
    r"$\mathcal L = \mathrm{CE}(X,y)"
    r" + \alpha\,\mathrm{CE}(\widetilde X,y)"
    r" + \lambda\,[1-\cos(z,\widetilde z)]$",
    ha="center", va="center", fontsize=10
)
ax.text(0.50, 0.125,
        r"$\alpha=1.0,\quad \lambda=0.15,\quad \sigma=0.4$",
        ha="center", va="center", fontsize=8)

arrow(0.87, 0.47, 0.72, 0.33)
arrow(0.87, 0.72, 0.72, 0.33)

ax.text(
    0.035, 0.93,
    "Component-targeted invariance for robust CSI-HAR",
    fontsize=12, fontweight="bold", va="center"
)
ax.text(
    0.035, 0.885,
    "The perturbation changes only the temporally stable component; "
    "the residual is preserved exactly by construction.",
    fontsize=8.5, va="center"
)

save_figure(fig, "Fig01_component_targeted_invariance", MAIN)

