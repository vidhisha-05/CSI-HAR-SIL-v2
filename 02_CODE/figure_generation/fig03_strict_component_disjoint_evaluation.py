"""
FIGURE 3
Strict component-disjoint evaluation.

Camera-ready version.

Panel A:
    Temporal-overlap audit -> grouping -> component-disjoint split

Panel B:
    Raw CE vs SIL-v2 under the original supplied split and
    component-disjoint evaluation.

All numerical values are read from:
07_FIGURE_DATA/camera_ready_figure_values.json
"""

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT
    / "07_FIGURE_DATA"
    / "camera_ready_figure_values.json"
)

OUT_DIR = (
    ROOT
    / "06_FIGURES"
    / "main"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD VERIFIED DATA
# ============================================================

# utf-8-sig handles both ordinary UTF-8 and PowerShell BOM.
with open(DATA_FILE, "r", encoding="utf-8-sig") as f:
    data = json.load(f)


original = data["original_split"]
strict = data["component_disjoint_single_split"]
audit = data["temporal_overlap_audit"]


# Original supplied split
original_raw = float(original["raw_ce"]["accuracy"])
original_sil = float(original["sil_v2"]["accuracy"])


# Component-disjoint split
strict_raw = float(strict["raw_ce"]["accuracy"])
strict_sil = float(strict["sil_v2"]["accuracy"])

delta_pp = float(strict["accuracy_difference_pp"])

ci_low = float(strict["bootstrap_ci_low_pp"])
ci_high = float(strict["bootstrap_ci_high_pp"])

p_value = float(strict["paired_permutation_p"])


# Temporal audit
total_windows = int(audit["total_windows"])
n_components = int(audit["components"])
cross_split_components = int(
    audit["cross_split_components"]
)
shared_blocks = int(
    audit["shared_exact_150_row_blocks_after_repartition"]
)


# ============================================================
# MATPLOTLIB STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,

    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,

    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,

    "legend.fontsize": 8.5,

    "axes.linewidth": 0.75,

    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# ============================================================
# FIGURE CANVAS
# ============================================================

fig = plt.figure(
    figsize=(8.2, 5.2)
)

gs = fig.add_gridspec(
    2,
    1,
    height_ratios=[1.0, 1.12],
    hspace=0.32,
)


# ============================================================
# PANEL A
# ============================================================

ax = fig.add_subplot(gs[0])

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


ax.text(
    0.0,
    1.02,
    "A  Evaluation protocol",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=11,
    fontweight="bold",
)


def box(
    x,
    y,
    w,
    h,
    title,
    lines,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.9,
        facecolor="white",
        edgecolor="black",
    )

    ax.add_patch(patch)

    ax.text(
        x + w / 2,
        y + h * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=8.7,
        fontweight="bold",
    )

    ax.text(
        x + w / 2,
        y + h * 0.34,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=7.6,
        linespacing=1.35,
    )


def arrow(
    x1,
    y1,
    x2,
    y2,
):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.8,
            color="black",
        )
    )


# Three-stage horizontal flow
box(
    0.025,
    0.30,
    0.255,
    0.42,
    "Supplied UT-HAR split",
    [
        f"{total_windows:,} windows",
        "250 CSI rows / window",
    ],
)

box(
    0.365,
    0.30,
    0.270,
    0.42,
    "Temporal overlap audit",
    [
        f"{n_components:,} overlap components",
        f"{cross_split_components:,} span supplied splits",
        "shared 150-row blocks",
    ],
)

box(
    0.720,
    0.30,
    0.255,
    0.42,
    "Component-disjoint split",
    [
        "Train / validation / test",
        f"{shared_blocks} shared exact blocks",
        "after repartition",
    ],
)


arrow(
    0.280,
    0.51,
    0.365,
    0.51,
)

arrow(
    0.635,
    0.51,
    0.720,
    0.51,
)


# Bottom protocol statement
ax.text(
    0.5,
    0.10,
    "Exact 150-row CSI blocks are not shared between the new partitions",
    ha="center",
    va="center",
    fontsize=8.2,
    fontweight="bold",
)


# ============================================================
# PANEL B
# ============================================================

ax = fig.add_subplot(gs[1])


ax.text(
    0.0,
    1.055,
    "B  Performance under the two evaluation protocols",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=11,
    fontweight="bold",
)


# ------------------------------------------------------------
# Coordinates
# ------------------------------------------------------------

protocol_x = np.array([0, 1])

raw_y = np.array([
    original_raw,
    strict_raw,
])

sil_y = np.array([
    original_sil,
    strict_sil,
])


# ------------------------------------------------------------
# Paired slope lines
# ------------------------------------------------------------

raw_line = ax.plot(
    protocol_x,
    raw_y,
    marker="o",
    markersize=7.5,
    linewidth=2.2,
    label="Raw CE",
)

sil_line = ax.plot(
    protocol_x,
    sil_y,
    marker="o",
    markersize=7.5,
    linewidth=2.2,
    label="SIL-v2",
)


# ------------------------------------------------------------
# Value labels
# ------------------------------------------------------------

ax.text(
    -0.035,
    original_raw + 0.45,
    f"{original_raw:.2f}%",
    ha="right",
    va="bottom",
    fontsize=8.5,
    fontweight="bold",
)

ax.text(
    -0.035,
    original_sil - 0.15,
    f"{original_sil:.2f}%",
    ha="right",
    va="top",
    fontsize=8.5,
    fontweight="bold",
)

ax.text(
    1.035,
    strict_raw - 0.15,
    f"{strict_raw:.2f}%",
    ha="left",
    va="top",
    fontsize=8.5,
    fontweight="bold",
)

ax.text(
    1.035,
    strict_sil + 0.45,
    f"{strict_sil:.2f}%",
    ha="left",
    va="bottom",
    fontsize=8.5,
    fontweight="bold",
)


# ------------------------------------------------------------
# Protocol labels
# ------------------------------------------------------------

ax.set_xticks(protocol_x)

ax.set_xticklabels(
    [
        "Original supplied\nsplit",
        "Component-disjoint\nsplit",
    ]
)


ax.set_ylabel(
    "Test accuracy (%)"
)


# ------------------------------------------------------------
# Y-axis
# ------------------------------------------------------------

ax.set_ylim(
    89.5,
    100.5,
)

ax.set_yticks(
    np.arange(
        90,
        101,
        2.5,
    )
)


# ------------------------------------------------------------
# Grid
# ------------------------------------------------------------

ax.grid(
    axis="y",
    linewidth=0.45,
    alpha=0.28,
)

ax.set_axisbelow(True)


# ------------------------------------------------------------
# Remove unnecessary borders
# ------------------------------------------------------------

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------

ax.legend(
    frameon=False,
    loc="upper right",
    ncol=2,
)


# ============================================================
# STRICT-SPLIT IMPROVEMENT ANNOTATION
# ============================================================

# Put annotation in open space between the two endpoints.
ax.annotate(
    f"+{delta_pp:.2f} pp",
    xy=(1.0, strict_sil),
    xytext=(0.73, 98.2),
    ha="center",
    va="center",
    fontsize=9,
    fontweight="bold",
    arrowprops=dict(
        arrowstyle="-|>",
        linewidth=0.9,
    ),
)


# ============================================================
# STATISTICAL FOOTNOTE
# ============================================================

fig.text(
    0.5,
    0.025,
    (
        f"Strict-split SIL-v2 advantage: +{delta_pp:.2f} pp  |  "
        f"paired bootstrap 95% CI "
        f"[+{ci_low:.2f}, +{ci_high:.2f}] pp  |  "
        f"paired permutation p = {p_value:.4f}"
    ),
    ha="center",
    va="bottom",
    fontsize=7.7,
)


# ============================================================
# SAVE
# ============================================================

fig.subplots_adjust(
    left=0.095,
    right=0.95,
    bottom=0.12,
    top=0.91,
)

output = (
    OUT_DIR
    / "Fig03_strict_component_disjoint_evaluation"
)


fig.savefig(
    output.with_suffix(".pdf"),
    dpi=600,
)

fig.savefig(
    output.with_suffix(".svg"),
    dpi=600,
)

fig.savefig(
    output.with_suffix(".png"),
    dpi=600,
)


plt.close(fig)


# ============================================================
# CONSOLE REPORT
# ============================================================

print("Figure 3 generated successfully.")
print()
print("Original supplied split:")
print(f"  Raw CE : {original_raw:.2f}%")
print(f"  SIL-v2 : {original_sil:.2f}%")
print()
print("Component-disjoint split:")
print(f"  Raw CE : {strict_raw:.2f}%")
print(f"  SIL-v2 : {strict_sil:.2f}%")
print(f"  Delta  : +{delta_pp:.2f} pp")
print()
print(
    f"Bootstrap CI : "
    f"[+{ci_low:.2f}, +{ci_high:.2f}] pp"
)
print(
    f"Permutation p: {p_value:.6f}"
)
print()
print("Saved:")
print(output.with_suffix(".pdf"))
print(output.with_suffix(".svg"))
print(output.with_suffix(".png"))

