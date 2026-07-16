"""
Fig. 4 — Ballistic simulation versus reported optical void-fraction references.
Standalone script. Loads precomputed main-configuration accepted results only.

Scientific boundary:
- The simulation curve is Cu ballistic morphology output.
- The 85° reference values are reported optical void fractions from ABEMA-style analysis.
- They are qualitative optical references/comparators, not direct experimental validation.
"""

import copy
import json
import pathlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.path as _mpath
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np


# ── matplotlib deepcopy patch ────────────────────────────────────────────────
def _fixed_path_deepcopy(self, memo):
    cls = type(self)
    r = cls.__new__(cls)
    memo[id(self)] = r
    r._vertices = copy.deepcopy(self._vertices, memo)
    r._codes = copy.deepcopy(self._codes, memo) if self._codes is not None else None
    r._interpolation_steps = self._interpolation_steps
    r._simplify_threshold = self._simplify_threshold
    r._should_simplify = self._should_simplify
    r._readonly = self._readonly
    return r

_mpath.Path.__deepcopy__ = _fixed_path_deepcopy


# ── paths ────────────────────────────────────────────────────────────────────
PROJECT      = pathlib.Path("/mnt/d/GLAD_PROJECT")
LEVEL_A_JSON = PROJECT / "03_VOXEL_DESCRIPTOR_ANALYSIS/analysis/levelA_results/levelA_accepted_results.json"
OUT = PROJECT / "05_THESIS_PAPER_ASSETS/figures/publication"
OUT.mkdir(parents=True, exist_ok=True)


# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  6.5,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.75",
    "lines.linewidth":  1.6,
    "axes.linewidth":   0.8,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "grid.linestyle":   "--",
    "grid.alpha":       0.30,
    "grid.linewidth":   0.5,
})

# Tunable: figure size in inches (single column)
SC = (3.65, 3.05)

# Tunable: annotation label font size — must be ≥ 6.5 for readability in print
LABEL_FONTSIZE = 7.0

# Tunable: offset of annotation text from the data point (in data units)
LABEL_OFFSET_X_LEFT  = 1.5    # how far left of 85° to place left-side labels
LABEL_OFFSET_X_RIGHT = 1.5    # how far right of 85° to place right-side labels


def savefig(fig, stem):
    # constrained_layout handles all spacing — do NOT call tight_layout here
    for ext in ("pdf", "png"):
        p = OUT / f"levelA_vs_literature_comparison.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  -> {p}  ({p.stat().st_size // 1024} kB)")
    plt.close(fig)


# ── load simulation data ─────────────────────────────────────────────────────
accepted = json.loads(LEVEL_A_JSON.read_text(encoding="utf-8"))
by_alpha = defaultdict(list)

for row in accepted:
    by_alpha[float(row["alpha"])].append(row)

sim_alphas = np.array(sorted(by_alpha.keys()))
sim_p = np.array([
    np.mean([float(r["porosity"]) for r in by_alpha[a]])
    for a in sim_alphas
])


# ── reported optical reference values at 85° ─────────────────────────────────
# NOTE (2026-06-17): the former Co/Al2O3 point (62.03%) was REMOVED. It did not
# appear in the manuscript text, broke the "cluster around 75%" description, and
# its provenance in the cited source (Fischer & Schubert 2013) could not be
# established locally. Only the three text-cited metal references are retained.
benchmarks = [
    {"label": "Co",              "alpha": 85, "void": 75.38, "err": 0.01},
    {"label": "Ti",              "alpha": 85, "void": 78.30, "err": 0.04},
    {"label": "NiFeMo",          "alpha": 85, "void": 75.0,  "err": 0.1},
]

bench_vals = [b["void"] for b in benchmarks]
band_min, band_max = min(bench_vals), max(bench_vals)


# ── figure: constrained_layout=True avoids tight_layout/bbox conflict ─────────
fig, ax = plt.subplots(figsize=SC, constrained_layout=True)

ax.plot(
    sim_alphas, sim_p,
    "o-", color="steelblue", lw=1.6, ms=5.5,
    mfc="white", mew=1.3, zorder=4
)

ax.axhspan(band_min, band_max, color="tomato", alpha=0.10, zorder=1)
ax.axvline(85, color="0.45", lw=0.8, ls="--", zorder=2)

for b in benchmarks:
    ax.errorbar(
        b["alpha"], b["void"], yerr=b["err"],
        fmt="o", color="tomato", ms=4.0,
        mfc="tomato", mec="tomato",
        elinewidth=0.75, capsize=2.0, zorder=6
    )

# Labels distributed left/right to avoid overlap with each other and the legend.
# Left side: Ti, Co (higher void); right side: NiFeMo, Co/Al2O3 (to separate).
# Tunable: xytext positions in data coordinates.
label_positions = {
    "Ti":              {"xytext": (85 - LABEL_OFFSET_X_LEFT, 79.2), "ha": "right"},
    "Co":              {"xytext": (85 - LABEL_OFFSET_X_LEFT, 76.2), "ha": "right"},
    "NiFeMo":          {"xytext": (85 + LABEL_OFFSET_X_RIGHT, 74.2), "ha": "left"},
}

for b in benchmarks:
    pos = label_positions[b["label"]]
    ax.annotate(
        b["label"],
        xy=(b["alpha"], b["void"]),
        xytext=pos["xytext"],
        textcoords="data",
        fontsize=LABEL_FONTSIZE,
        color="tomato",
        ha=pos["ha"], va="center",
        arrowprops=dict(
            arrowstyle="-", color="tomato",
            lw=0.55, shrinkA=0.5, shrinkB=2.5
        ),
        zorder=7
    )

ax.set_xlabel(r"Deposition angle $\alpha$ (°)")
ax.set_ylabel("Void fraction (%)")
ax.set_xlim(57, 91)
ax.set_ylim(72, 101)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.grid(True)

legend_handles = [
    Line2D([0], [0], color="steelblue", marker="o", markersize=5.0,
           markerfacecolor="white", markeredgewidth=1.1, lw=1.5,
           label="Ballistic simulation (Cu)"),
    Line2D([0], [0], color="tomato", marker="o", markersize=4.0,
           markerfacecolor="tomato", markeredgecolor="tomato", lw=0,
           label=r"Optical reference values at $\alpha=85^\circ$"),
    mpatches.Patch(facecolor="tomato", edgecolor="none", alpha=0.10,
                   label="Reference-value span"),
    Line2D([0], [0], color="0.45", lw=0.8, ls="--",
           label=r"Reference angle, $\alpha=85^\circ$"),
]

# Legend placed above the axes; constrained_layout reserves the space automatically.
ax.legend(
    handles=legend_handles,
    loc="lower center", bbox_to_anchor=(0.5, 1.02),
    ncol=2, frameon=True, fontsize=6.2,
    borderpad=0.35, labelspacing=0.35,
    handlelength=1.6, handletextpad=0.5, columnspacing=0.9
)

savefig(fig, "fig04_optical_reference_values_85deg")
