"""
Appendix figure — finite-size (box-width L) convergence study, app:boxconv.
Standalone script. Reads DATA_TRUSTED from the verified box-convergence
data module only; no simulation, no re-fitting.
"""
import pathlib
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BOXCONV_DIR = pathlib.Path(
    "/mnt/d/GLAD_PROJECT/01_GLAD_SIMULATION/simulation_batch/runs/"
    "P1_PITCH_BOX_ALIASING_V1/analysis"
)
sys.path.insert(0, str(BOXCONV_DIR))
from boxconv_data import DATA_TRUSTED  # noqa: E402

OUT = pathlib.Path("/mnt/d/GLAD_PROJECT/05_THESIS_PAPER_ASSETS/figures/publication")

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "grid.linestyle": "--", "grid.alpha": 0.3, "grid.linewidth": 0.5,
})

by_h = {}
for d in DATA_TRUSTED:
    by_h.setdefault(d.h, []).append(d)

colors = {25: "#4c72b0", 50: "#55a868", 100: "#c44e52"}
markers = {25: "s", 50: "^", 100: "o"}

fig, ax = plt.subplots(figsize=(4.2, 3.4), constrained_layout=True)

for h in sorted(by_h.keys()):
    pts = by_h[h]
    by_L = {}
    for d in pts:
        by_L.setdefault(d.L, []).append(d.density)
    Ls = sorted(by_L.keys())
    means = [np.mean(by_L[L]) for L in Ls]
    ax.plot(Ls, means, marker=markers[h], color=colors[h], lw=1.3, ms=5.5,
            mfc="white", mew=1.2, label=rf"$h={int(h)}$ nm")
    # scatter individual seed replicates for L values with n>1
    for L in Ls:
        vals = by_L[L]
        if len(vals) > 1:
            ax.scatter([L] * len(vals), vals, color=colors[h], s=10, alpha=0.5, zorder=5)

ax.set_xlabel(r"Box width $L_x = L_y$ (nm)")
ax.set_ylabel(r"Bulk atom density (atoms/nm$^3$)")
ax.grid(True)
ax.legend(loc="upper right", frameon=True)

for ext in ("pdf", "png"):
    p = OUT / f"boxconv_density_vs_L.{ext}"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  -> {p}  ({p.stat().st_size // 1024} kB)")
plt.close(fig)
