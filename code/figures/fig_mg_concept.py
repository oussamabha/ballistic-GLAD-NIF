"""
Fig -- Maxwell-Garnett effective-medium concept schematic (conceptual
diagram, no simulation data). Illustrates aligned high-index cylindrical
inclusions (columns) in a low-index host (void), and why the ordinary vs
extraordinary optical axes differ (form birefringence), for the
Maxwell-Garnett subsubsection in ssec:nif_methods.
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle

OUT = pathlib.Path("/mnt/d/GLAD_PROJECT/05_THESIS_PAPER_ASSETS/figures/publication")

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.titlesize": 8.3,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.6, 3.0), constrained_layout=True)

# Panel A: real geometry -> averaged into effective medium
axA.set_xlim(0, 10)
axA.set_ylim(-1.1, 8)
axA.set_aspect("equal")
axA.axis("off")
axA.set_title(r"(a) Real geometry $\to$ effective medium", fontsize=8.3)

axA.add_patch(Rectangle((0.3, 1.1), 3.6, 6.5, facecolor="#f7f7f5", edgecolor="#999", linewidth=0.8))
np.random.seed(5)
for cx in [1.1, 2.0, 2.9]:
    for cy in np.arange(1.7, 7.1, 1.0):
        jitter = np.random.uniform(-0.08, 0.08)
        axA.add_patch(Ellipse((cx + jitter, cy), 0.55, 0.85, facecolor="#a85a1c", edgecolor="none", alpha=0.85))
axA.text(2.1, 0.55, "columns (high index)\nin void (host)", ha="center", fontsize=6.6, color="#555")

axA.add_patch(FancyArrowPatch((4.3, 4.6), (5.5, 4.6), arrowstyle="-|>", mutation_scale=12, color="#333", lw=1.3))
axA.text(4.9, 4.95, "MG\naverage", ha="center", fontsize=6.3, color="#333")

axA.add_patch(Rectangle((5.9, 1.1), 3.6, 6.5, facecolor="#e7ede9", edgecolor="#3f7a6c", linewidth=0.8))
axA.text(7.7, 4.3, r"$\varepsilon_o,\ \varepsilon_e$", ha="center", va="center", fontsize=13, color="#3f7a6c")
axA.text(7.7, 0.55, "uniaxial effective medium\n(anisotropic)", ha="center", fontsize=6.6, color="#3f7a6c")

# Panel B: ordinary vs extraordinary axis
axB.set_xlim(0, 10)
axB.set_ylim(-1.1, 8.6)
axB.set_aspect("equal")
axB.axis("off")
axB.set_title(r"(b) Ordinary vs extraordinary response", fontsize=8.3)

axB.add_patch(Rectangle((2.0, 1.2), 6.0, 6.4, facecolor="#e7ede9", edgecolor="#3f7a6c", linewidth=0.8))
for cx in np.linspace(2.8, 7.2, 5):
    axB.plot([cx, cx], [1.6, 7.2], color="#a85a1c", lw=2.2, alpha=0.7, solid_capstyle="round")

# E parallel to columns (extraordinary)
axB.add_patch(FancyArrowPatch((1.0, 4.3), (1.0, 6.2), arrowstyle="-|>", mutation_scale=11, color="#333", lw=1.2))
axB.text(0.7, 5.2, r"$E_e \parallel$" "\ncolumns", ha="right", fontsize=6.6, color="#333")

# E perpendicular to columns (ordinary)
axB.add_patch(FancyArrowPatch((5.0, 8.1), (7.0, 8.1), arrowstyle="-|>", mutation_scale=11, color="#333", lw=1.2))
axB.text(6.0, 8.45, r"$E_o \perp$ columns", ha="center", fontsize=6.6, color="#333")

axB.text(5.0, 0.55, r"$\Delta n = n_e - n_o$: form birefringence" "\n"
                     r"from the geometric anisotropy alone",
         ha="center", fontsize=6.6, color="#555")

for ext in ("pdf", "png"):
    p = OUT / f"maxwell_garnett_concept.{ext}"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  -> {p}  ({p.stat().st_size // 1024} kB)")
plt.close(fig)
