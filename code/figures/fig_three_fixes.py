"""
Fig -- Schematic summary of the three remedies tested in ssec:disc_tier2
(Fix A: richer spatial encoding, Fix B: loss/target reframing, Fix C:
partial-realization conditioning) and why each fails for a different
reason. Conceptual diagram, no simulation data; the underlying numeric
results are real and already reported in the text and
Table tab:fixc_permtest.
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


def box(ax, x, y, w, h, text, fc, ec, fontsize=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                 facecolor=fc, edgecolor=ec, linewidth=1.1, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=3)


fig, ax = plt.subplots(figsize=(7.2, 3.3), constrained_layout=True)
ax.set_xlim(0, 14)
ax.set_ylim(2.2, 11)
ax.axis("off")

ax.text(0.2, 10.6, "Baseline problem: pooled hard-label occupancy prediction is trivial\n"
                     r"at $\alpha=80°,85°$",
        fontsize=7.6, color="#333")

rows = [
    (8.3, "Fix A — richer spatial encoding", "#3f7a6c",
     "Multi-resolution hash-grid encoder\n(replaces Fourier positional encoding)",
     r"No improvement at any angle" "\n" r"($\Delta\mathrm{IoU}\approx0.0001$)",
     "More local capacity does not help:\nthe missing signal isn't a capacity problem."),
    (5.4, "Fix B — loss/target reframing", "#a85a1c",
     "Continuous density regression\n(MSE, no binarisation during training)",
     r"No improvement" "\n" r"($\Delta\mathrm{IoU}=0.0001,0.0003$)",
     "The BCE/soft-label mismatch alone\nis not the binding constraint."),
    (2.5, "Fix C — partial-realization\nconditioning", "#555",
     "Condition prediction on the SAME\nrealization's own observed region",
     "Fails permutation test\nat both $n=3$ and $n=8$\n(see summary table)",
     "Real seed-specific signal exists in the\ninput but the encoder never learns to use it."),
]

for y0, title, color, method, result, reason in rows:
    box(ax, 0.2, y0, 3.5, 1.9, title, "#f2f2f0", color, fontsize=7.4)
    box(ax, 4.1, y0 + 0.15, 3.6, 1.6, method, "#eef1ef", "#888", fontsize=6.6)
    ax.add_patch(FancyArrowPatch((3.7, y0 + 0.95), (4.1, y0 + 0.95), arrowstyle="-|>",
                                  mutation_scale=10, color="#555", lw=1.0))
    box(ax, 8.1, y0 + 0.15, 2.5, 1.6, result, "#fdeee7", "#a85a1c", fontsize=6.6)
    ax.add_patch(FancyArrowPatch((7.7, y0 + 0.95), (8.1, y0 + 0.95), arrowstyle="-|>",
                                  mutation_scale=10, color="#555", lw=1.0))
    ax.text(11.0, y0 + 0.95, reason, ha="left", va="center", fontsize=6.5, color="#333")

for ext in ("pdf", "png"):
    p = OUT / f"three_fixes_summary.{ext}"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  -> {p}  ({p.stat().st_size // 1024} kB)")
plt.close(fig)
