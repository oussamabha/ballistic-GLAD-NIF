"""
Fig -- Distribution of soft-label occupancy values rho, per deposition angle.
Standalone script. Reads the archived train_samples.npz dataset only
(same file used by reproduce_alpha_z_oracle.py / Table tab:label_stats
in P2). No training, no simulation -- direct visual proof of the
"90-98% of labels lie in the ambiguous (0.1,0.9) range" claim.
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = pathlib.Path("/mnt/d/GLAD_PROJECT/02_PINN_NIF/pinn_training/datasets/levelA_helical_nif/train_samples.npz")
OUT = pathlib.Path("/mnt/d/GLAD_PROJECT/05_THESIS_PAPER_ASSETS/figures/publication")

ALPHA_NORM_TO_DEG = {
    -1.0: 60.0, -0.655: 65.0, -0.310: 70.0, 0.034: 75.0,
    0.379: 80.0, 0.724: 85.0, 0.862: 87.0, 1.0: 89.0,
}

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        9,
    "axes.titlesize":   8.5,
    "axes.labelsize":   9,
    "xtick.labelsize":  7.5,
    "ytick.labelsize":  7.5,
    "axes.linewidth":   0.8,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

train = np.load(DATA)["data"]
alpha_col = train[:, 3]
rho = train[:, 4]

keys = np.array(sorted(ALPHA_NORM_TO_DEG.keys()))
idx = np.abs(alpha_col[:, None] - keys[None, :]).argmin(axis=1)
alpha_deg = np.array([ALPHA_NORM_TO_DEG[keys[i]] for i in idx])

angles = sorted(ALPHA_NORM_TO_DEG.values())
fig, axes = plt.subplots(1, 8, figsize=(10.2, 2.1), sharey=True, constrained_layout=True)

bins = np.linspace(0, 1, 41)
for ax, a in zip(axes, angles):
    r = rho[alpha_deg == a]
    f_soft = ((r > 0.1) & (r < 0.9)).mean() * 100
    ax.hist(r, bins=bins, color="#3f7a6c", edgecolor="none")
    ax.axvspan(0.1, 0.9, color="#a85a1c", alpha=0.10, zorder=0)
    ax.set_title(rf"{a}°" + "\n" + rf"$f_{{\rm soft}}$={f_soft:.0f}%", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yscale("log")

axes[0].set_ylabel("Voxel count")
fig.supxlabel(r"Soft occupancy label $\rho$", fontsize=9)

for ext in ("pdf", "png"):
    p = OUT / f"soft_label_histogram.{ext}"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  -> {p}  ({p.stat().st_size // 1024} kB)")
plt.close(fig)
