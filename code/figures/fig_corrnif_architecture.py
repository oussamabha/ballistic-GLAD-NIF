"""
Fig -- CorrNIF architecture schematic (conceptual diagram, no simulation
data). Box/arrow diagram of the FiLM-conditioned coordinate network
described in ssec:nif_methods (CorrNIF architecture and training):
(alpha_norm, r_norm) -> Fourier features -> 3 FiLM-conditioned hidden
layers -> h_g(r;alpha).
"""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = pathlib.Path("/mnt/d/GLAD_PROJECT/05_THESIS_PAPER_ASSETS/figures/publication")

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def box(ax, x, y, w, h, text, fc, ec, fontsize=7.6, textcolor="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                 facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=textcolor, zorder=3)


def arrow(ax, p0, p1, color="#555"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10,
                                  color=color, linewidth=1.1, zorder=1))


fig, ax = plt.subplots(figsize=(7.0, 2.55), constrained_layout=True)
ax.set_xlim(0, 14)
ax.set_ylim(0.3, 4.9)
ax.axis("off")

# Inputs
box(ax, 0.3, 3.6, 1.9, 0.9, r"$r_{\rm norm}$" "\n(radial distance)", "#eef1ef", "#3f7a6c")
box(ax, 0.3, 1.6, 1.9, 0.9, r"$\alpha_{\rm norm}$" "\n(deposition angle)", "#f4ece2", "#a85a1c")

# Fourier features
box(ax, 2.7, 3.6, 2.0, 0.9, "Random Fourier\nfeatures ($n$=12)", "#eef1ef", "#3f7a6c")
arrow(ax, (2.2, 4.05), (2.7, 4.05))

# FiLM MLP branch
box(ax, 2.7, 1.6, 2.0, 0.9, "FiLM MLP\n" r"$\alpha \to (\gamma,\beta)$", "#f4ece2", "#a85a1c")
arrow(ax, (2.2, 2.05), (2.7, 2.05))

# 3 hidden layers
layer_x = [5.5, 7.7, 9.9]
for i, lx in enumerate(layer_x):
    box(ax, lx, 3.0, 1.7, 1.4,
        f"Hidden layer {i+1}\nwidth 64, SiLU\n" r"$\gamma_i h+\beta_i$",
        "#e7ede9", "#333333", fontsize=6.8)
    if i == 0:
        arrow(ax, (4.7, 4.05), (lx, 3.7))
    else:
        arrow(ax, (layer_x[i-1] + 1.7, 3.7), (lx, 3.7))
    # FiLM arrows from beta box up into each layer
    arrow(ax, (3.7 + i * 0.05, 2.5), (lx + 0.4, 3.0), color="#a85a1c")

# Output
box(ax, 12.0, 3.35, 1.8, 0.9, r"$h_g(r;\alpha)$" "\n(predicted)", "#fdf2ec", "#a85a1c")
arrow(ax, (11.6, 3.7), (12.0, 3.8))

ax.text(7.0, 0.75,
        r"FiLM: each hidden layer's activation is rescaled by $\gamma_i(\alpha)$ and shifted by "
        r"$\beta_i(\alpha)$" "\n"
        r"before the nonlinearity — the network's function of $r$ changes continuously with $\alpha$,"
        "\n"
        "the same way a parametric dispersion model is evaluated at any wavelength without refitting.",
        ha="center", va="center", fontsize=6.8, color="#555")

for ext in ("pdf", "png"):
    p = OUT / f"corrnif_architecture.{ext}"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  -> {p}  ({p.stat().st_size // 1024} kB)")
plt.close(fig)
