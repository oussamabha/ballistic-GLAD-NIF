"""
Fig. 3 — PCA morphology space.
Standalone script. Loads precomputed descriptors only.
No simulation, no training, no descriptor extraction.

Uses "apparent width-gradient proxy" instead of "fanning angle".
"""

import copy
import json
import pathlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.path as _mpath
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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
PROJECT  = pathlib.Path("/mnt/d/GLAD_PROJECT")
DESCR_JSON = PROJECT / "03_VOXEL_DESCRIPTOR_ANALYSIS/analysis/levelA_results/levelA_descriptors.json"
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
    "legend.fontsize":  8,
    "axes.linewidth":   0.8,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "grid.linestyle":   "--",
    "grid.alpha":       0.30,
    "grid.linewidth":   0.5,
})

# Tunable: figure width × height in inches (double column)
DC = (7.20, 3.10)

# Tunable: colorbar geometry
CBAR_FRACTION = 0.038
CBAR_PAD      = 0.018
CBAR_SHRINK   = 0.92

# Tunable: panel gap (only affects constrained_layout internal padding)
W_PAD = 0.10
H_PAD = 0.06


def savefig(fig, stem):
    # constrained_layout handles spacing — do NOT call tight_layout here
    p = OUT / "morphology_pca_space_levelA.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  -> {p}  ({p.stat().st_size // 1024} kB)")
    plt.close(fig)


# ── load data ────────────────────────────────────────────────────────────────
descr = json.loads(DESCR_JSON.read_text(encoding="utf-8"))

ANOMALOUS = {"LA_helical_alpha_300nm_alpha060_seed000_300nm_helical_pitch150"}

valid_results = [
    r for r in descr["results"]
    if r["job_id"] not in ANOMALOUS
]

pca_rows = []
for row in valid_results:
    pca_rows.append({
        "alpha":          row["alpha_deg"],
        "seed":           row["seed"],
        "openness":       row["openness_proxy"],
        "filling":        row["filling_fraction"],
        "anisotropy":     row["anisotropy_ratio"],
        "width_gradient": row.get("fanning_angle_deg") or 0.0,
    })

X = np.array([
    [r["openness"], r["filling"], r["anisotropy"], r["width_gradient"]]
    for r in pca_rows
])

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pc = pca.fit_transform(X_scaled)
var     = pca.explained_variance_ratio_ * 100
load_pc1 = pca.components_[0]

alpha_values = sorted({r["alpha"] for r in pca_rows})
cmap = matplotlib.colormaps.get_cmap("viridis")
norm = plt.Normalize(vmin=min(alpha_values), vmax=max(alpha_values))


# ── figure with constrained_layout ───────────────────────────────────────────
# NOTE: do NOT pass wspace/hspace inside gridspec_kw when constrained_layout=True
# — those keys are silently ignored and can produce inconsistent spacing.
# Use set_constrained_layout_pads() instead.
fig, axes = plt.subplots(
    1, 2,
    figsize=DC,
    constrained_layout=True,
    gridspec_kw={"width_ratios": [1.00, 1.00]},
)

fig.set_constrained_layout_pads(
    w_pad=W_PAD, h_pad=H_PAD, wspace=0.00, hspace=0.00
)

ax_pca, ax_load = axes


# ── Panel A: PCA morphology space ────────────────────────────────────────────
for i, row in enumerate(pca_rows):
    ax_pca.scatter(
        X_pc[i, 0], X_pc[i, 1],
        color=cmap(norm(row["alpha"])),
        s=42, edgecolor="white", linewidth=0.4, zorder=4
    )

seed0_idx    = [i for i, r in enumerate(pca_rows) if r["seed"] == 0]
seed0_sorted = sorted(seed0_idx, key=lambda i: pca_rows[i]["alpha"])

xs = X_pc[seed0_sorted, 0]
ys = X_pc[seed0_sorted, 1]

ax_pca.plot(xs, ys, "k--", lw=0.8, alpha=0.55, zorder=2)
ax_pca.annotate(
    "",
    xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.0)
)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(
    sm, ax=ax_pca,
    fraction=CBAR_FRACTION, pad=CBAR_PAD, shrink=CBAR_SHRINK
)
cbar.set_label(r"Deposition angle $\alpha$ (°)", fontsize=8)
cbar.set_ticks(alpha_values)
cbar.set_ticklabels([f"{a}°" for a in alpha_values])   # tick labels were missing before
cbar.ax.tick_params(labelsize=7)

ax_pca.set_title("(a) Morphology space")
ax_pca.set_xlabel(f"PC1 ({var[0]:.1f}% variance)")
ax_pca.set_ylabel(f"PC2 ({var[1]:.1f}% variance)")
ax_pca.grid(True)


# ── Panel B: PC1 feature loadings ────────────────────────────────────────────
feature_labels = [
    "Openness proxy",
    "Filling fraction",
    "XY anisotropy",
    "Width-gradient proxy",   # single-line to avoid label clipping
]

bar_colors = ["steelblue" if v >= 0 else "tomato" for v in load_pc1]

ax_load.barh(feature_labels, load_pc1,
             color=bar_colors, edgecolor="white", linewidth=0.5)
ax_load.axvline(0, color="black", lw=0.7)
ax_load.set_title("(b) PC1 feature loadings")
ax_load.set_xlabel("PC1 loading")
ax_load.grid(True, axis="x")
ax_load.tick_params(axis="y", labelsize=7.5)

savefig(fig, "fig03_morphology_pca_space")
