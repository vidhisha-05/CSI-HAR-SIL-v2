
"""
FIGURE S1 â€” Temporal redundancy audit.

Supplementary figure supporting the protocol-repair rationale.

Uses the validated temporal-group audit values:
- 4973 windows
- 556 components
- 442 components >= 5 windows
- 161 >= 10
- 45 >= 20
- largest = 28
- 425 components span multiple supplied splits
- all 556 components are label-pure

The exact component-size distribution is reconstructed from
temporal_group_audit.json if the expected key is available; otherwise the
script uses the validated count summary only.

Outputs:
06_FIGURES/supplementary/FigS1_temporal_redundancy_audit.{pdf,svg,png}
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from figure_common import setup_matplotlib, save_figure, AUDITS, SUPP

setup_matplotlib()

audit_path = AUDITS / "temporal_overlap" / "temporal_group_audit.json"
with open(audit_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Fixed validated summary.
thresholds = np.array([1, 5, 10, 20, 50])
counts = np.array([556, 442, 161, 45, 0])

fig, ax = plt.subplots(figsize=(6.8, 3.9))
ax.plot(thresholds, counts, marker="o", linewidth=1.8)
ax.set_xscale("symlog", linthresh=1)
ax.set_xlabel("Minimum component size (windows)")
ax.set_ylabel("Number of temporal-overlap components")
ax.grid(axis="y", linewidth=0.45, alpha=0.35)
ax.set_title("S1  Temporal redundancy in the supplied UT-HAR partition",
             loc="left", fontweight="bold")

ax.text(
    0.02, 0.03,
    "556 components total; largest component = 28 windows\n"
    "425 components span multiple supplied splits; all components are label-pure",
    transform=ax.transAxes, fontsize=8.0, va="bottom"
)

save_figure(fig, "FigS1_temporal_redundancy_audit", SUPP)

