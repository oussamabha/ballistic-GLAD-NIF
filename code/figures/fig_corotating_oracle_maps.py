"""
Fig — Corotating-frame mean occupancy maps M_train(x_rot, y_rot).
Standalone script. Reads precomputed voxel arrays only, no training.

Reuses the exact derotation logic of
02_PINN_NIF/pinn_training/oracle_corotating_ceiling.py (nearest-neighbour
lookup, same convention) to regenerate M_train for the two extreme angles
(alpha=65 deg, lowest peak/mean; alpha=89 deg, highest peak/mean) discussed
in ssec:nif_voxel, and plots them side by side to visually support the
"even a strongly peaked template gives zero oracle IoU gain" claim.
"""
import math
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HELIX_PITCH_NM    = 150.0
FILM_THICKNESS_NM = 300.0
BINARY_THR        = 0.5

PROJECT_ROOT = pathlib.Path("/mnt/d/GLAD_PROJECT")
VOXEL_DIR    = PROJECT_ROOT / "02_PINN_NIF/pinn_training/datasets/universal_glad_pipeline/voxels"
OUT          = PROJECT_ROOT / "05_THESIS_PAPER_ASSETS/figures/publication"


def phi(k, nz):
    pitch_voxels = HELIX_PITCH_NM / (FILM_THICKNESS_NM / nz)
    return 2.0 * math.pi * k / pitch_voxels


def derotate_slice(sl, angle):
    nx, ny = sl.shape
    cos_a, sin_a = math.cos(-angle), math.sin(-angle)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    i_src = cos_a * (ii - cx) - sin_a * (jj - cy) + cx
    j_src = sin_a * (ii - cx) + cos_a * (jj - cy) + cy
    i_src = np.clip(np.round(i_src).astype(int), 0, nx - 1)
    j_src = np.clip(np.round(j_src).astype(int), 0, ny - 1)
    return sl[i_src, j_src]


def build_M_train(alpha):
    vf = next(VOXEL_DIR.glob(f"rho_LA_helical_alpha_300nm_alpha{alpha:03d}_seed000*.npy"))
    arr = np.load(vf).astype(np.float32)
    binary = arr > BINARY_THR
    nx, ny, nz = binary.shape
    k_split = round(nz * 0.80)
    M_sum = np.zeros((nx, ny), dtype=np.float64)
    angles = phi(np.arange(k_split), nz)
    for k in range(k_split):
        M_sum += derotate_slice(binary[:, :, k].astype(np.float64), angles[k])
    M_train = (M_sum / k_split).astype(np.float32)
    peak_mean = float(M_train.max() / (M_train.mean() + 1e-9))
    return M_train, peak_mean


plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

ALPHAS_TO_SHOW = [65, 89]
fig = plt.figure(figsize=(9.2, 3.05), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0])
ax_schem = fig.add_subplot(gs[0, 0])
ax_data = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]

# --- Panel (a): schematic of the derotation procedure (conceptual, no data) ---
ax_schem.set_xlim(0, 10)
ax_schem.set_ylim(0, 10)
ax_schem.set_aspect("equal")
ax_schem.axis("off")
ax_schem.set_title("(a) Derotation procedure", fontsize=8.5)

import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle as _Circle

# stack of rotated z-slices (side view), each rotated by phi(k)
for i, (yy, ang_deg) in enumerate(zip([7.6, 6.3, 5.0], [15, 55, 100])):
    w, h = 2.3, 0.85
    t = mpatches.Rectangle((0.4, yy - h / 2), w, h, angle=ang_deg,
                            rotation_point=(0.4, yy), facecolor="#a85a1c",
                            edgecolor="none", alpha=0.55 + 0.15 * i)
    ax_schem.add_patch(t)
ax_schem.text(0.3, 9.1, r"slices at different $z$" "\n" r"(phase $\phi(z)$)", fontsize=6.4, color="#555")

ax_schem.add_patch(FancyArrowPatch((3.2, 6.3), (5.6, 6.3), arrowstyle="-|>",
                                    mutation_scale=12, color="#333", lw=1.3))
ax_schem.text(4.4, 6.75, "derotate\nby $-\\phi(z)$", ha="center", fontsize=6.3, color="#333")

# common corotating frame: aligned slices stacked -> averaged
for i, yy in enumerate([3.6, 2.9, 2.2]):
    t = mpatches.Rectangle((6.0, yy - 0.32), 2.3, 0.5, facecolor="#3f7a6c",
                            edgecolor="none", alpha=0.5 + 0.15 * i)
    ax_schem.add_patch(t)
ax_schem.add_patch(FancyArrowPatch((7.15, 1.85), (7.15, 0.85), arrowstyle="-|>",
                                    mutation_scale=11, color="#3f7a6c", lw=1.2))
ax_schem.text(7.15, 0.45, r"average over $z$" "\n" r"$\to M_{\mathrm{train}}(x_{\mathrm{rot}},y_{\mathrm{rot}})$",
              ha="center", fontsize=6.3, color="#3f7a6c")
ax_schem.text(7.15, 4.35, "common\ncorotating frame", ha="center", fontsize=6.4, color="#555")

# --- Panels (b),(c): real M_train data ---
ims = []
for ax, alpha, label in zip(ax_data, ALPHAS_TO_SHOW, ["b", "c"]):
    M_train, peak_mean = build_M_train(alpha)
    im = ax.imshow(M_train.T, origin="lower", cmap="viridis", vmin=0, vmax=1,
                    extent=[0, M_train.shape[0], 0, M_train.shape[1]])
    ims.append(im)
    ax.set_title(rf"({label}) $\alpha={alpha}^\circ$  (peak/mean $={peak_mean:.2f}$)", fontsize=8.5)
    ax.set_xlabel(r"$x_{\mathrm{rot}}$ (voxel)")
    ax.set_yticks([])

ax_data[0].set_ylabel(r"$y_{\mathrm{rot}}$ (voxel)")
ax_data[0].set_yticks(ax_data[0].get_xticks())

cbar = fig.colorbar(ims[-1], ax=ax_data, fraction=0.045, pad=0.03)
cbar.set_label(r"Mean derotated occupancy $M_{\mathrm{train}}$")

for ext in ("pdf", "png"):
    p = OUT / f"corotating_oracle_maps.{ext}"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  -> {p}  ({p.stat().st_size // 1024} kB)")
plt.close(fig)
