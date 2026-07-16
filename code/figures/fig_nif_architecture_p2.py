"""
Fig -- P2 shared FiLM-conditioned NIF architecture schematic (Experiments
2-3), conceptual diagram, no simulation data. Box/arrow diagram matching
the description in ssec:arch: (x_n,y_n,z_n,alpha_n) -> Fourier encoding
(n=32) -> 4 FiLM-conditioned hidden layers (width 256) -> sigmoid rho_hat.
"""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = pathlib.Path("/mnt/d/GLAD_PROJECT/05_THESIS_PAPER_ASSETS/figures/publication")

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def box(ax, x, y, w, h, text, fc, ec, fontsize=7.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                 facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=3)


def arrow(ax, p0, p1, color="#555"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10,
                                  color=color, linewidth=1.1, zorder=1))


fig, ax = plt.subplots(figsize=(7.2, 2.9), constrained_layout=True)
ax.set_xlim(0, 15)
ax.set_ylim(0.2, 5.3)
ax.axis("off")

box(ax, 0.2, 3.55, 2.1, 1.0, r"$(x_{\rm n},y_{\rm n},z_{\rm n})$" "\nspatial coordinate", "#eef1ef", "#3f7a6c")
box(ax, 0.2, 1.5, 2.1, 1.0, r"$\alpha_{\rm n}$" "\ndeposition angle", "#f4ece2", "#a85a1c")

box(ax, 2.7, 3.55, 2.1, 1.0, "Fourier encoding\n" r"$n_{\rm freq}=32$", "#eef1ef", "#3f7a6c")
arrow(ax, (2.3, 4.05), (2.7, 4.05))

box(ax, 2.7, 1.5, 2.1, 1.0, "FiLM MLP\n" r"$\alpha_{\rm n}\to(\gamma,\beta)$", "#f4ece2", "#a85a1c")
arrow(ax, (2.3, 2.0), (2.7, 2.0))

layer_x = [5.7, 7.7, 9.7, 11.7]
for i, lx in enumerate(layer_x):
    box(ax, lx, 3.15, 1.75, 1.7,
        f"Layer {i+1}\nwidth 256\n" r"$\gamma_i h+\beta_i$",
        "#e7ede9", "#333333", fontsize=6.8)
    if i == 0:
        arrow(ax, (4.8, 4.05), (lx, 4.0))
    else:
        arrow(ax, (layer_x[i-1] + 1.75, 4.0), (lx, 4.0))
    arrow(ax, (3.8 + i * 0.02, 2.4), (lx + 0.4, 3.15), color="#a85a1c")

box(ax, 13.7, 3.55, 1.1, 1.0, r"$\hat{\rho}$" "\n" r"$\in[0,1]$", "#fdf2ec", "#a85a1c", fontsize=8.5)
arrow(ax, (13.45, 4.0), (13.7, 4.05))

ax.text(7.5, 0.65,
        r"Same shared backbone for Experiment 2 (soft label $\hat{\rho}\approx\rho_{\rm vox}$) and "
        r"Experiment 3 (hard label $\hat{\rho}\in\{0,1\}$)"
        "\n"
        "— only the training target and label binarisation change; architecture and FiLM mechanism are identical.",
        ha="center", va="center", fontsize=6.8, color="#555")

for ext in ("pdf", "png"):
    p = OUT / f"nif_architecture_p2.{ext}"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  -> {p}  ({p.stat().st_size // 1024} kB)")
plt.close(fig)
