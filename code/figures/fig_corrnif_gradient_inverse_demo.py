"""
Fig — CorrNIF gradient-based inverse-design demonstration.
Standalone script. Loads precomputed corrnif_gradient_inverse_results.json only.

Scientific intent:
- Left panel: brute-force grid loss landscape L(alpha) with the 5 gradient-
  descent trajectories overlaid, showing all restarts (from deliberately
  wrong starting angles) converge to the same minimum as the exhaustive
  grid search.
- Right panel: per-restart loss-vs-iteration convergence curves.
- Demonstrates the differentiability advantage claimed for CorrNIF: alpha
  recovered by gradient descent through the frozen, trained model, not by
  re-running the simulator or a discrete grid search.
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# This script lives at <repo_root>/code/figures/.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS   = REPO_ROOT / "data/small_artifacts/corrnif_gradient_inverse_results.json"
OUT       = REPO_ROOT / "data/small_artifacts"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "axes.linewidth":  0.8,
    "lines.linewidth": 1.6,
    "figure.dpi":      300,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "grid.linestyle":  "--",
    "grid.alpha":      0.30,
    "grid.linewidth":  0.5,
})

with open(RESULTS) as f:
    d = json.load(f)

grid_alphas = np.array(d["grid_alphas_deg"])
grid_losses = np.array(d["grid_losses"])
restarts    = d["restarts"]
target      = d["target_alpha_deg"]
grid_best   = d["grid_search_best_alpha_deg"]
rec_mean    = d["recovered_alpha_mean_deg"]

cmap = cm.viridis
norm = plt.Normalize(vmin=0, vmax=len(restarts) - 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

# ── left: loss landscape + gradient trajectories ─────────────────────────────
ax1.semilogy(grid_alphas, grid_losses + 1e-12, color="0.35", lw=1.2, zorder=2,
             label="Brute-force grid, 0.005$^\\circ$ res.")
for i, r in enumerate(restarts):
    traj_a = np.array(r["trajectory_alpha"])
    traj_l = np.array(r["trajectory_loss"]) + 1e-12
    ax1.plot(traj_a, traj_l, color=cmap(norm(i)), lw=1.0, alpha=0.85, zorder=3,
             marker="o", markevery=[0], markersize=3.5,
             label=f"init={r['init_alpha_deg']:.0f}$^\\circ$")
ax1.axvline(target, color="crimson", lw=1.0, ls=":", zorder=1,
            label=f"true $\\alpha$={target}$^\\circ$")
ax1.axvline(grid_best, color="0.15", lw=0.8, ls="--", zorder=1,
            label=f"grid optimum={grid_best:.2f}$^\\circ$")
ax1.set_xlabel(r"deposition angle $\alpha$ ($^{\circ}$)")
ax1.set_ylabel(r"MSE loss (log scale)")
ax1.set_title("(a) loss landscape + gradient trajectories")
ax1.legend(fontsize=5.6, loc="upper right", ncol=1, framealpha=0.9)

# ── right: loss vs iteration convergence ─────────────────────────────────────
for i, r in enumerate(restarts):
    traj_l = np.array(r["trajectory_loss"]) + 1e-12
    ax2.semilogy(traj_l, color=cmap(norm(i)), lw=1.2,
                 label=f"init={r['init_alpha_deg']:.0f}$^\\circ$")
ax2.set_xlabel("gradient-descent step")
ax2.set_ylabel("MSE loss (log scale)")
ax2.set_title("(b) convergence from 5 restarts")
ax2.legend(fontsize=6, loc="upper right")

fig.tight_layout()
p = OUT / "corrnif_gradient_inverse_demo.pdf"
fig.savefig(p)
print(f"  -> {p}  ({p.stat().st_size // 1024} kB)")
plt.close(fig)
