"""
Fig -- Cross-sectional density maps of the simulated ballistic helical
Cu GLAD film: direct visual "proof" of the columnar shadowing structure
the rest of the paper characterises quantitatively.
Standalone script. Reads precomputed 2nm voxel density fields and the
frozen topology_sweep_all27_RESULTS.json only. No simulation, no training.
Restyled from generate_morphology_figure_v2_annotated.py to match the
journal-clean convention used elsewhere in this paper (no in-figure
suptitle, no floating annotation box -- that information goes in the
LaTeX caption).
"""
import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VOXDIR = pathlib.Path(
    "/mnt/d/GLAD_PROJECT/00_PHD_KNOWLEDGE_HUB/99_PENDING_REVIEW/"
    "P1_CLEAN_HOMOGENEOUS_PUBLICATION_CAMPAIGN_20260619/P1_VOXELS"
)
OUT = pathlib.Path("/mnt/d/GLAD_PROJECT/05_THESIS_PAPER_ASSETS/figures/publication")
TOPOLOGY_JSON = pathlib.Path("/mnt/d/GLAD_PROJECT/05_THESIS_PAPER_ASSETS/figures/topology_sweep_all27_RESULTS.json")

VOX_NM = 2.0
BOX_NM = 100.0

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "axes.linewidth":   0.8,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})


def load(alpha_str, seed):
    fname = VOXDIR / f"rho_P1_alpha{alpha_str}_seed{seed:03d}_300nm_pitch150.npy"
    return np.load(str(fname))


def get_lf(alpha, seed):
    data = json.loads(TOPOLOGY_JSON.read_text())
    for row in data["rows"]:
        if row["alpha"] == alpha and row["seed"] == seed:
            return row["largest_frac"], row["occupancy"]
    raise ValueError(f"no row for alpha={alpha}, seed={seed}")


v89 = load("089", 1)
v60 = load("060", 1)
lf_89, occ_89 = get_lf(89, 1)
lf_60, occ_60 = get_lf(60, 1)

z_mid_idx = round(150.0 / VOX_NM)
y_mid_idx = round(50.0 / VOX_NM)

panel_A = v89[:, :, z_mid_idx].T
panel_B = v89[:, y_mid_idx, :].T
panel_C = v60[:, :, z_mid_idx].T

cmap = "viridis"
vmin, vmax = 0.0, 1.0

fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5), constrained_layout=True)

ax = axes[0]
im = ax.imshow(panel_A, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
               extent=[0, BOX_NM, 0, BOX_NM], aspect="equal")
ax.set_xlabel("x (nm)")
ax.set_ylabel("y (nm)")
ax.set_title(r"(a) $\alpha=89°$, $xy$ at $z=150$ nm", fontsize=8.5)

ax = axes[1]
im = ax.imshow(panel_B, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
               extent=[0, BOX_NM, 0, v89.shape[2] * VOX_NM], aspect="auto")
ax.set_xlabel("x (nm)")
ax.set_ylabel("z (nm)")
ax.set_title(r"(b) $\alpha=89°$, $xz$ at $y=50$ nm", fontsize=8.5)

ax = axes[2]
im = ax.imshow(panel_C, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
               extent=[0, BOX_NM, 0, BOX_NM], aspect="equal")
ax.set_xlabel("x (nm)")
ax.set_ylabel("y (nm)")
ax.set_title(r"(c) $\alpha=60°$, $xy$ at $z=150$ nm", fontsize=8.5)

cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.035, pad=0.02)
cbar.set_label("Normalised voxel density")

for ext in ("pdf", "png"):
    p = OUT / f"morphology_crosssection.{ext}"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  -> {p}  ({p.stat().st_size // 1024} kB)")
plt.close(fig)

print(f"alpha=89 seed=1: occupied={occ_89*100:.1f}%, void={(1-occ_89)*100:.1f}%, LF={lf_89:.3f}")
print(f"alpha=60 seed=1: occupied={occ_60*100:.1f}%, void={(1-occ_60)*100:.1f}%, LF={lf_60:.3f}")
