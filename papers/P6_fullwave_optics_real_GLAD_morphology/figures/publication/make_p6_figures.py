#!/usr/bin/env python3
"""P6 manuscript figures. Numbers copied verbatim from the manuscript's own
Table 1 (Mueller elements) and Table 2 (angle sweep) -- no new computation,
just visualizing already-reported results.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent
plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.linewidth": 0.8})

# ------------------------------------------------------- Fig 1: Mueller elements
elements = ["mm12", "mm21", "mm33", "mm34", "mm22"]
measured = [-0.658, -0.669, 0.325, 0.598, 0.943]
single = [-0.898, -0.898, 0.300, -0.321, 1.000]
avg3 = [-0.876, -0.876, 0.382, -0.291, 1.000]

x = np.arange(len(elements))
w = 0.27
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.bar(x - w, measured, w, color="#231C13", label="measured")
ax.bar(x, single, w, color="#9B8D78", label="single realisation")
ax.bar(x + w, avg3, w, color="#AC581F", label="3-seed average")
ax.axhline(0, color="black", lw=0.6)
ax.set_xticks(x)
ax.set_xticklabels(elements)
ax.set_ylabel("Mueller element value")
ax.set_title("Realisation averaging moves 3 of 4 adjustable\nelements toward measured; mm22 cannot move", fontsize=9.5)
ax.legend(fontsize=8, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "P6_mueller_comparison.pdf")
fig.savefig(OUT / "P6_mueller_comparison.png", dpi=300)
plt.close(fig)
print("Fig 1 written")

# ------------------------------------------------------- Fig 2: angle sweep
alpha = [65, 70, 72, 75, 80, 85, 87, 89]
void = [50.9, 58.0, 65.4, 65.1, 80.2, 90.8, 92.9, 98.7]
n_re = [0.818, 0.777, 0.735, 0.736, 0.672, 0.750, 0.798, 0.962]
n_im = [1.602, 1.442, 1.258, 1.265, 0.786, 0.329, 0.238, 0.036]

fig, ax1 = plt.subplots(figsize=(5.2, 3.4))
ax1.plot(alpha, void, "o-", color="#AC581F", lw=1.6, ms=6, label="void fraction")
ax1.set_xlabel("Deposition angle $\\alpha$ (deg)")
ax1.set_ylabel("Void fraction (%)", color="#AC581F")
ax1.tick_params(axis="y", labelcolor="#AC581F")

ax2 = ax1.twinx()
ax2.plot(alpha, n_im, "s--", color="#2C7A70", lw=1.4, ms=5, label="$n_{eff}$ (imag)")
ax2.set_ylabel("$n_{eff}$ (imaginary part)", color="#2C7A70")
ax2.tick_params(axis="y", labelcolor="#2C7A70")

ax1.set_title("Void fraction rises, extinction falls,\nwith deposition angle (α=60° excluded, voxel artefact)", fontsize=9.5)
fig.tight_layout()
fig.savefig(OUT / "P6_angle_sweep.pdf")
fig.savefig(OUT / "P6_angle_sweep.png", dpi=300)
plt.close(fig)
print("Fig 2 written")
