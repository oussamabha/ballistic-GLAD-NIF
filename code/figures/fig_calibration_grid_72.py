"""
Fig -- Calibration grid vs blind test point schematic (conceptual, no
simulation data). Shows which angles CorrNIF was trained on and which
one (72 deg) was held out entirely, for ssec:nif_prospective.
"""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path("/mnt/d/GLAD_PROJECT/05_THESIS_PAPER_ASSETS/figures/publication")

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

train_angles = [65, 70, 75, 80, 85, 87, 89]
test_angle = 72

fig, ax = plt.subplots(figsize=(6.0, 1.5), constrained_layout=True)
ax.hlines(0, 60, 92, color="#888", lw=1.2, zorder=1)
ax.plot(train_angles, [0] * len(train_angles), "o", color="#3f7a6c", ms=11,
        mec="black", mew=0.8, zorder=3, label="Training angle (seen)")
ax.plot([test_angle], [0], "*", color="#a85a1c", ms=22, mec="black", mew=0.8,
        zorder=4, label=r"Held out: $\alpha=72°$ (never trained on)")

for a in train_angles:
    ax.text(a, 0.32, f"{a}°", ha="center", fontsize=7.8, color="#3f7a6c")
ax.text(test_angle, -0.42, "72°", ha="center", fontsize=8.2, color="#a85a1c", fontweight="bold")

ax.set_xlim(60, 92)
ax.set_ylim(-0.7, 0.7)
ax.axis("off")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.55), ncol=2, fontsize=7.3,
          frameon=False, handletextpad=0.4, columnspacing=1.3)

for ext in ("pdf", "png"):
    p = OUT / f"calibration_grid_72.{ext}"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  -> {p}  ({p.stat().st_size // 1024} kB)")
plt.close(fig)
