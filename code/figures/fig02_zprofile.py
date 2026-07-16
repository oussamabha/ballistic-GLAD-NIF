"""
Fig. 2 — Depth-resolved void fraction (Z-profiles), absolute and normalised.
Standalone script. Loads precomputed descriptors and voxel arrays only.
No simulation, no training, no descriptor extraction.
"""

import copy
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
DESCR_JSON = PROJECT / "03_VOXEL_DESCRIPTOR_ANALYSIS/analysis/levelA_results/levelA_descriptors.json"
VOXEL_DIR  = PROJECT / "02_PINN_NIF/pinn_training/datasets/universal_glad_pipeline/voxels"
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
    "lines.linewidth":  1.3,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "grid.linestyle":   "--",
    "grid.alpha":       0.30,
    "grid.linewidth":   0.5,
})

# Tunable: figure size in inches (double column)
DC = (6.69, 3.80)

# Tunable: colorbar geometry
CBAR_FRACTION = 0.035
CBAR_PAD      = 0.03


def savefig(fig, stem):
    # constrained_layout handles spacing — do NOT call tight_layout here
    for ext in ("pdf", "png"):
        p = OUT / f"levelA_zprofile.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  -> {p}  ({p.stat().st_size // 1024} kB)")
    plt.close(fig)


# ── load data ────────────────────────────────────────────────────────────────
descr = json.loads(DESCR_JSON.read_text(encoding="utf-8"))
ANOMALOUS = {"LA_helical_alpha_300nm_alpha060_seed000_300nm_helical_pitch150"}

seen_a = set()
selected = []
for r in sorted(descr["results"], key=lambda x: (x["alpha_deg"], x["seed"])):
    a = r["alpha_deg"]
    if r["job_id"] in ANOMALOUS:
        continue
    if a not in seen_a:
        seen_a.add(a)
        selected.append(r)

NZ = 151
VOX_Z = 300.0 / (NZ - 1)
Z_NM = np.arange(NZ) * VOX_Z

alphas_plot = sorted(set(r["alpha_deg"] for r in selected))
n = len(alphas_plot)
cmap_plasma = matplotlib.colormaps.get_cmap("plasma").resampled(n)
a2col = {a: cmap_plasma(i) for i, a in enumerate(alphas_plot)}


# ── plot ─────────────────────────────────────────────────────────────────────
# sharey=True: both panels plot the same quantity on the same 0-105% scale, so
# the y-axis (and its label) is shown only once, on the left panel. The original
# bug set an identical ylabel on BOTH panels with sharey=False — panel (b)'s
# rotated label text then collided with panel (a)'s right edge.
fig, axes = plt.subplots(
    1, 2, figsize=DC, sharey=True,
    constrained_layout=True,
    gridspec_kw={"width_ratios": [1.00, 1.00]},
)
ax_abs, ax_norm = axes

for r in selected:
    a = r["alpha_deg"]
    vpath = VOXEL_DIR / r["voxel_file"]
    if not vpath.exists():
        continue
    rho = np.load(vpath).astype(np.float32)
    void_z = 1.0 - rho.mean(axis=(0, 1))
    occ = np.where(void_z < 0.97)[0]
    h_top = float(Z_NM[occ[-1]]) if occ.size > 0 else 300.0
    col = a2col[a]
    ax_abs.plot(Z_NM, void_z * 100, color=col, lw=1.3)
    ax_norm.plot(Z_NM / h_top, void_z * 100, color=col, lw=1.3)

ax_abs.set_ylabel(r"Layer void fraction $1 - \bar{\rho}_z$ (%)")
for ax in axes:
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.grid(True)
# tick labels on the shared right panel are redundant with the left panel
ax_norm.tick_params(labelleft=False)

ax_abs.set_xlabel(r"Depth from substrate $z$ (nm)")
ax_abs.set_xlim(0, 305)
ax_abs.xaxis.set_major_locator(ticker.MultipleLocator(50))
ax_abs.set_title("(a) Absolute depth", fontsize=9)

ax_norm.set_xlabel(r"Normalised height $z / h$")
ax_norm.set_xlim(0, 1.05)
ax_norm.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax_norm.set_title("(b) Normalised height", fontsize=9)

# Colorbar shared across both panels
sm = plt.cm.ScalarMappable(
    cmap=cmap_plasma,
    norm=plt.Normalize(vmin=min(alphas_plot), vmax=max(alphas_plot))
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes.tolist(), pad=CBAR_PAD, fraction=CBAR_FRACTION)
cbar.set_label(r"Deposition angle $\alpha$ (°)", fontsize=8)
cbar.set_ticks(alphas_plot)
cbar.set_ticklabels([f"{int(a)}°" for a in alphas_plot])
cbar.ax.tick_params(labelsize=7)

savefig(fig, "fig02_levelA_zprofile")
