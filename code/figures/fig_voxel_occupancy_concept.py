"""
Fig -- Conceptual diagram of the voxel occupancy prediction task.
Not simulation data: a small illustrative grid contrasting the "easy"
binary case (clean occupied/empty boundary) with the "hard" ambiguous
case (most voxels intermediate), which is exactly the situation
quantified for real data in Table tab:label_stats / Figure fig:soft_hist.
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path("/mnt/d/GLAD_PROJECT/05_THESIS_PAPER_ASSETS/figures/publication")

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.titlesize": 8.3,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

np.random.seed(7)
n = 10

# Panel A: clean binary case -- a sharp column boundary
clean = np.zeros((n, n))
for j in range(n):
    col_top = 3 + int(2 * np.sin(j * 0.9) > 0.5) * 0
clean[:6, :] = 1.0
clean[6:, :] = 0.0
# add a jittered but still-sharp boundary
for j in range(n):
    shift = np.random.choice([-1, 0, 0, 1])
    row = 6 + shift
    clean[:row, j] = 1.0
    clean[row:, j] = 0.0

# Panel B: ambiguous case -- most voxels intermediate (0.1-0.9)
ambig = np.clip(0.5 + 0.28 * np.random.randn(n, n), 0.02, 0.98)
# keep a few clearly empty corners for realism
ambig[0, 0] = 0.03
ambig[0, -1] = 0.04
ambig[-1, 0] = 0.05

fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1), constrained_layout=True)

cmap = "viridis"
for ax, arr, title, frac_label in zip(
    axes, [clean, ambig],
    [r"(a) Sharp boundary $\to$ learnable", r"(b) Mostly ambiguous $\to$ nothing to learn"],
    ["most voxels near 0 or 1", "most voxels near 0.5"]
):
    im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, origin="lower")
    ax.set_title(title, fontsize=8.3)
    ax.set_xticks([]); ax.set_yticks([])
    for (i, j), v in np.ndenumerate(arr):
        txt_color = "white" if v < 0.5 else "black"
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5.6, color=txt_color, alpha=0.85)
    ax.set_xlabel(frac_label, fontsize=7.5)

cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
cbar.set_label(r"Occupancy value $\rho_{\rm vox}$: 0 = empty, 1 = occupied")

for ext in ("pdf", "png"):
    p = OUT / f"voxel_occupancy_concept.{ext}"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  -> {p}  ({p.stat().st_size // 1024} kB)")
plt.close(fig)
