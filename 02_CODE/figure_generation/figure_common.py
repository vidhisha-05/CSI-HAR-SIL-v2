
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------
# Camera-ready figure configuration
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "03_RESULTS"
AUDITS = ROOT / "04_AUDITS"
FIGURES = ROOT / "06_FIGURES"
FIGURE_DATA = ROOT / "07_FIGURE_DATA"

MAIN = FIGURES / "main"
SUPP = FIGURES / "supplementary"
MAIN.mkdir(parents=True, exist_ok=True)
SUPP.mkdir(parents=True, exist_ok=True)

DPI = 600
FONT_FAMILY = "DejaVu Sans"

def setup_matplotlib():
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })

def save_figure(fig, stem, folder=MAIN):
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(folder / f"{stem}.{ext}", dpi=DPI)
    plt.close(fig)

def pct(x):
    return f"{x:.1f}%"

def pp(x):
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.1f} pp"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

