"""
Fig. 5 — Normalised pair correlation h_g(r; alpha) with near-field inset.
Standalone script. Loads precomputed corr_functions.npz only.

Scientific intent:
- Main plot: global pair-correlation decay over r = 0-25 voxels.
- Inset: exact near-field region r = 0-5 voxels.
- Manual connectors show that the inset corresponds to the near-field region.
- This is a morphology-statistics figure, not experimental validation.
"""

import copy
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.path as _mpath
from matplotlib.patches import ConnectionPatch
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
PROJECT  = pathlib.Path("/mnt/d/GLAD_PROJECT")
NPZ_FILE = PROJECT / "02_PINN_NIF/pinn_training/runs/morphology_pipeline/corr_functions.npz"
OUT = PROJECT / "05_THESIS_PAPER_ASSETS/figures/publication"
OUT.mkdir(parents=True, exist_ok=True)


# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "axes.linewidth":  0.8,
    "lines.linewidth": 1.6,
    "figure.dpi":      300,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "grid.linestyle":  "--",
    "grid.alpha":      0.30,
    "grid.linewidth":  0.5,
})

# Tunable: figure size in inches (single-column wide)
SC_WIDE = (4.20, 3.10)

# Tunable: inset geometry [left, bottom, width, height] in axes fraction.
# Keep width ≤ 0.44 so the inset clears the colorbar on the right.
INSET_RECT   = [0.37, 0.42, 0.44, 0.46]

# Tunable: colorbar geometry
CBAR_FRACTION = 0.045
CBAR_PAD      = 0.025


def savefig(fig, stem):
    # Single-axis layout: bbox_inches="tight" alone is sufficient.
    # Do NOT also call tight_layout() — the two conflict and double-squeeze margins.
    p = OUT / "nif_pair_correlation.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  -> {p}  ({p.stat().st_size // 1024} kB)")
    plt.close(fig)


# ── load data ────────────────────────────────────────────────────────────────
npz = np.load(NPZ_FILE)

alphas = [int(a) for a in npz["alphas"]]
h_all  = {int(a): npz["h_r"][i]    for i, a in enumerate(npz["alphas"])}
r_all  = {int(a): npz["r_bins"][i] for i, a in enumerate(npz["alphas"])}

cmap = cm.viridis
norm = plt.Normalize(vmin=min(alphas), vmax=max(alphas))


# ── plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=SC_WIDE)

for alpha in alphas:
    ax.plot(r_all[alpha], h_all[alpha], color=cmap(norm(alpha)), lw=1.6, zorder=2)

# Dashed rectangle marks the near-field region in the main plot
nearfield_rect = patches.Rectangle(
    (0, -0.05), 5, 1.15,
    linewidth=0.9, edgecolor="0.45",
    facecolor="none", linestyle="--", zorder=4
)
ax.add_patch(nearfield_rect)

ax.set_xlabel(r"Radial separation $r$ (voxels)")
ax.set_ylabel(r"Pair correlation $h_g(r;\alpha)$")
ax.set_xlim(0, 25)
ax.set_ylim(-0.05, 1.10)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.grid(True)

# ── Inset: r = 0–5 vox ───────────────────────────────────────────────────────
axins = ax.inset_axes(INSET_RECT)

for alpha in alphas:
    r = r_all[alpha]
    h = h_all[alpha]
    mask = r <= 5.0
    axins.plot(r[mask], h[mask], color=cmap(norm(alpha)), lw=1.5, zorder=2)

axins.set_xlim(0, 5)
axins.set_ylim(-0.05, 1.05)
axins.set_xlabel(r"$r$ (vox)", fontsize=7)
axins.set_ylabel(r"$h_g$", fontsize=7)
axins.set_title(r"Near-field ($r \leq 5$ vox)", fontsize=6.5)  # shortened to avoid collision
axins.tick_params(labelsize=6)
axins.xaxis.set_major_locator(ticker.MultipleLocator(1))
axins.grid(True, alpha=0.30)
axins.set_facecolor("white")
axins.patch.set_alpha(0.96)

# ── Connectors: near-field rectangle corners → inset corners ─────────────────
# con_left:  left  edge of rectangle (x=0, y=0 data) → bottom-LEFT  of inset
# con_right: right edge of rectangle (x=5, y=0 data) → bottom-RIGHT of inset
# Bug in original: both connectors used xyB=(0, 0) — right connector now uses (1, 0).
con_left = ConnectionPatch(
    xyA=(0, 0), coordsA=ax.transData,
    xyB=(0, 0), coordsB=axins.transAxes,
    axesA=ax, axesB=axins,
    color="0.45", lw=0.8, linestyle="-", zorder=5, clip_on=False
)
con_right = ConnectionPatch(
    xyA=(5, 0), coordsA=ax.transData,
    xyB=(1, 0), coordsB=axins.transAxes,   # was (0,0) — fixed to right corner
    axesA=ax, axesB=axins,
    color="0.45", lw=0.8, linestyle="-", zorder=5, clip_on=False
)
fig.add_artist(con_left)
fig.add_artist(con_right)

# ── Colorbar ──────────────────────────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=CBAR_PAD, fraction=CBAR_FRACTION)
cbar.set_label(r"Deposition angle $\alpha$ (°)", fontsize=8)
cbar.set_ticks(alphas)
cbar.set_ticklabels([f"{a}°" for a in alphas])
cbar.ax.tick_params(labelsize=7)

savefig(fig, "fig05_pair_correlation_nearfield")
