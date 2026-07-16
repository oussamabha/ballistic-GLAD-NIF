"""
Fig -- Schematic of the ballistic helical GLAD deposition process.
Conceptual diagram (no simulation data), built with matplotlib patches only.
Explains the mechanism referenced throughout Methods sec:methods:
collimated beam at angle alpha, first-contact sticking with no surface
diffusion, self-shadowing, and substrate rotation tracing a helix.
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, Arc
import matplotlib.patches as mpatches

OUT = pathlib.Path("/mnt/d/GLAD_PROJECT/05_THESIS_PAPER_ASSETS/figures/publication")

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.titlesize": 8.5,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 3.1), constrained_layout=True)

# ---------------- Panel (a): side view, shadowing mechanism ----------------
axL.set_xlim(0, 10)
axL.set_ylim(0, 8)
axL.set_aspect("equal")
axL.axis("off")
axL.set_title(r"(a) Ballistic shadowing, side view", fontsize=8.5)

axL.add_patch(Rectangle((0.5, 0.3), 9, 0.5, facecolor="#c9c2b4", edgecolor="black", linewidth=0.8, zorder=2))
axL.text(5, 0.05, "substrate", ha="center", fontsize=7.5, color="#555")

alpha_deg = 65
alpha = np.radians(alpha_deg)
col_x = [2.0, 4.3, 6.6]
np.random.seed(3)
col_h = [2.3, 3.4, 2.0]
beta = np.radians(38)  # column tilt angle < alpha (tangent-rule-like, illustrative)
for cx, ch in zip(col_x, col_h):
    n_beads = int(ch * 4)
    for k in range(n_beads):
        frac = k / n_beads
        by = 0.8 + frac * ch
        bx = cx + frac * ch * np.tan(beta)
        axL.add_patch(Circle((bx, by), 0.17, facecolor="#3f7a6c", edgecolor="none", alpha=0.85, zorder=3))

# shadow region behind the tallest column
sx, sh = col_x[1], col_h[1]
shadow_dx = sh / np.tan(alpha)
axL.fill_between([sx + 0.3, sx + 0.3 + shadow_dx], 0.8, sh + 0.8,
                  color="#a85a1c", alpha=0.12, zorder=1)
axL.text(sx + 0.3 + shadow_dx * 0.5, 0.55, "shadow\n(no arrivals)", ha="center",
         fontsize=6.3, color="#a85a1c")

# incoming beam arrows
for bx0 in [1.0, 3.3, 5.6, 7.9]:
    by0 = 7.4
    bx1 = bx0 + 1.6 * np.sin(alpha) * 0.9
    by1 = by0 - 1.6 * np.cos(alpha) * 0.9
    axL.add_patch(FancyArrowPatch((bx0, by0), (bx1, by1), arrowstyle="-|>",
                                   mutation_scale=9, color="#555", linewidth=0.9, zorder=1))

axL.annotate("", xy=(3.3, 6.7), xytext=(3.3, 7.9),
             arrowprops=dict(arrowstyle="-", color="#555", lw=0.6, linestyle="--"))
arc = Arc((3.3, 7.4), 1.4, 1.4, angle=0, theta1=270, theta2=270 + alpha_deg, color="#555", lw=0.8)
axL.add_patch(arc)
axL.text(3.75, 7.15, r"$\alpha$", fontsize=9, color="#333")
axL.text(6.6, 7.6, "collimated\natom beam", fontsize=7, ha="left", color="#555")
axL.text(2.0, 5.0, "column grows\ntoward the source", fontsize=6.3, ha="center", color="#3f7a6c")

# ---------------- Panel (b): top view, substrate rotation ----------------
axR.set_xlim(-1.6, 1.6)
axR.set_ylim(-1.6, 1.6)
axR.set_aspect("equal")
axR.axis("off")
axR.set_title(r"(b) Substrate rotation, top view", fontsize=8.5)

axR.add_patch(Circle((0, 0), 1.15, facecolor="#c9c2b4", edgecolor="black", linewidth=0.8, zorder=1))
theta = np.linspace(0.3, 2 * np.pi - 0.3, 200)
spiral_r = 0.08 + 0.9 * (theta / theta.max())
axR.plot(spiral_r * np.cos(theta), spiral_r * np.sin(theta), color="#3f7a6c", lw=1.4, zorder=2)
axR.add_patch(FancyArrowPatch((spiral_r[-2] * np.cos(theta[-2]), spiral_r[-2] * np.sin(theta[-2])),
                               (spiral_r[-1] * np.cos(theta[-1]), spiral_r[-1] * np.sin(theta[-1])),
                               arrowstyle="-|>", mutation_scale=10, color="#3f7a6c", linewidth=1.4, zorder=2))
axR.add_patch(FancyArrowPatch((0, 1.35), (0.35, 1.32), connectionstyle="arc3,rad=0.4",
                               arrowstyle="-|>", mutation_scale=9, color="#555", linewidth=0.9))
axR.text(0.55, 1.35, r"programmed rotation" "\n" r"($2\pi$ per pitch $p$)",
         fontsize=6.5, color="#555", ha="left", va="center")
axR.text(0, -1.45, "substrate seen from above\n(fixed beam direction)", fontsize=6.5,
         ha="center", color="#555")

for ext in ("pdf", "png"):
    p = OUT / f"deposition_schematic.{ext}"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  -> {p}  ({p.stat().st_size // 1024} kB)")
plt.close(fig)
