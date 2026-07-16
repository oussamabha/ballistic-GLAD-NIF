"""
Fig. 7 — CorrNIF prospective validation at alpha = 72 degrees.
Standalone script. Loads alpha72_prospective_results.json and corr_functions.npz.

Uses "reference simulation", not "ground truth".
"""

import copy
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.path as _mpath
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
PROJECT    = pathlib.Path("/mnt/d/GLAD_PROJECT")
NPZ_FILE   = PROJECT / "02_PINN_NIF/pinn_training/runs/morphology_pipeline/corr_functions.npz"
PROSP_JSON = PROJECT / "02_PINN_NIF/pinn_training/runs/morphology_pipeline/alpha72_prospective_results.json"
OUT = PROJECT / "05_THESIS_PAPER_ASSETS/figures/publication"
OUT.mkdir(parents=True, exist_ok=True)


# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   6,
    "legend.framealpha": 0.90,
    "legend.edgecolor":  "0.75",
    "axes.linewidth":    0.8,
    "lines.linewidth":   1.6,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "grid.linestyle":    "--",
    "grid.alpha":        0.30,
    "grid.linewidth":    0.5,
})

# Tunable: figure size in inches (single-column wide)
SC_WIDE = (4.10, 3.00)

# Tunable: near-field annotation position (in data coordinates).
# Arrow tip points into the shaded region; text placed outside to the right to
# avoid overlap with curves (which are at h_g ≈ 0.8–1.0 at small r values).
NF_ARROW_TIP  = (5.0, 0.60)   # right edge of shaded region at mid-height
NF_TEXT_POS   = (7.0, 0.82)   # text anchor, where curves have decayed


def savefig(fig, stem):
    # Single-axis layout: bbox_inches="tight" is sufficient.
    # Do NOT also call tight_layout() — the two conflict and double-squeeze margins.
    p = OUT / "nif_alpha72_prospective.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  -> {p}  ({p.stat().st_size // 1024} kB)")
    plt.close(fig)


# ── load data ────────────────────────────────────────────────────────────────
prosp = json.loads(PROSP_JSON.read_text(encoding="utf-8"))

r          = np.array(prosp["r_bins"])
h_ref      = np.array(prosp["h_gt72"])
h_corrnif  = np.array(prosp["h_corrnif"])
h_linear   = np.array(prosp["h_linear_blend"])
h_nn       = np.array(prosp["h_nn_alpha70"])

r2_corrnif = float(prosp["r2_corrnif"])
r2_linear  = float(prosp["r2_linear"])
r2_nn      = float(prosp["r2_nn70"])

npz    = np.load(NPZ_FILE)
alphas = [int(a) for a in npz["alphas"]]
h_all  = {int(a): npz["h_r"][i]    for i, a in enumerate(npz["alphas"])}
r_all  = {int(a): npz["r_bins"][i] for i, a in enumerate(npz["alphas"])}

cmap = cm.viridis
norm = plt.Normalize(vmin=min(alphas), vmax=max(alphas))


# ── plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=SC_WIDE)

# Context curves — faint background; excluded from legend to keep it compact
for alpha in alphas:
    ax.plot(r_all[alpha], h_all[alpha],
            color=cmap(norm(alpha)), lw=0.7, alpha=0.25, zorder=1)

# Nearest training neighbours — slightly highlighted
ax.plot(r_all[70], h_all[70], color=cmap(norm(70)), lw=1.4, alpha=0.80,
        label="70° (training nbr.)", zorder=2)
ax.plot(r_all[75], h_all[75], color=cmap(norm(75)), lw=1.4, alpha=0.80,
        label="75° (training nbr.)", zorder=2)

# Reference and model predictions — main visual hierarchy
ax.plot(r, h_ref,     color="black",       lw=2.1,
        label="Reference simulation (72°)", zorder=6)
ax.plot(r, h_corrnif, color="crimson",     lw=1.8, ls="--",
        label=f"CorrNIF ($R^2$={r2_corrnif:.3f})", zorder=7)   # 3 decimals: readable in legend
ax.plot(r, h_linear,  color="royalblue",   lw=1.7, ls=":",
        label=f"Linear blend ($R^2$={r2_linear:.3f})", zorder=5)
ax.plot(r, h_nn,      color="darkmagenta", lw=1.5, ls="-.",
        label=f"Nearest nbr. ($R^2$={r2_nn:.3f})", zorder=4)

# Near-field shading
ax.axvspan(2, 5, color="gold", alpha=0.08, zorder=0)

# Near-field label: placed to the RIGHT of the shaded zone with an arrow pointing
# left into it. Avoids overlap with curves, which are at h_g ≈ 0.8–1.0 at r < 5.
ax.annotate(
    "near-field\n$2 \\leq r \\leq 5$ vox",
    xy=NF_ARROW_TIP,
    xytext=NF_TEXT_POS,
    fontsize=5.5, ha="left", va="center", color="0.45",
    arrowprops=dict(arrowstyle="-", color="0.45", lw=0.6, shrinkA=1, shrinkB=1)
)

ax.set_xlabel(r"Radial separation $r$ (voxels)")
ax.set_ylabel(r"$h_g(r;\,72^\circ)$")
ax.set_xlim(0, 25)
ax.set_ylim(-0.05, 1.10)
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.grid(True)

# Legend: upper-right is empty at large r (curves decay to ~0 by r = 20).
# Compact params: handlelength, borderpad, labelspacing keep the box small.
ax.legend(
    loc="upper right",
    frameon=True,
    fontsize=6,
    handlelength=1.8,
    borderpad=0.40,
    labelspacing=0.28,
    handletextpad=0.45,
)

savefig(fig, "fig07_alpha72_prospective_validation")
