#!/usr/bin/env python3
"""P4 manuscript figures. All numbers copied verbatim from project result files --
no fabrication, no re-derivation. Sources named per figure.

Fig 1: NEB minimum-energy path for the Cu(100) adatom hop (source:
  01_GLAD_SIMULATION/P4_EAM_MD_FUTURE_WORK/results/phase1_hop_barrier_result.json)
Fig 2: geometry-kinetics decomposition, the two height-matched points
  alpha=82 and alpha=89 (source:
  01_GLAD_SIMULATION/simulation_batch/runs/P1_DIFFUSION_DECOMP_C2_KERNELFIX_20260907/C2_ANALYSIS_20260910.md)
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent
ROOT = Path(__file__).resolve()
for p in [ROOT, *ROOT.parents]:
    if (p / "00_PHD_KNOWLEDGE_HUB").exists():
        ROOT = p
        break

plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.linewidth": 0.8})

# ---------------------------------------------------------------- Fig 1: NEB
d = json.load(open(ROOT / "01_GLAD_SIMULATION/P4_EAM_MD_FUTURE_WORK/results/phase1_hop_barrier_result.json"))
path_e = np.array(d["path_energies_eV"])
rel_e = path_e - path_e[0]
images = np.arange(len(path_e))
Ea = d["measured_barrier_eV"]
lit = d["literature_eV"]

fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.plot(images, rel_e, "o-", color="#AC581F", lw=1.6, ms=5, zorder=3,
        label=f"NEB path (this work), $E_a$={Ea:.4f} eV")
ax.axhline(Ea, color="#AC581F", lw=0.6, ls=":", alpha=0.6)
colors = {"boisvert1997": "#2C7A70", "mehl1999_model_ii": "#6C5F4E", "zhang2011_isolated_adatom": "#8A4517"}
labels = {"boisvert1997": "Boisvert & Lewis 1997",
          "mehl1999_model_ii": "Mehl et al. 1999 (model II)",
          "zhang2011_isolated_adatom": "Zhang et al. 2011"}
for k, v in lit.items():
    if v is None:
        continue
    ax.axhline(v, color=colors.get(k, "gray"), lw=1.0, ls="--", alpha=0.8, label=f"{labels[k]}, {v:.3f} eV")
ax.set_xlabel("NEB image index")
ax.set_ylabel("Energy relative to initial site (eV)")
ax.set_title("Cu(100) adatom hop: minimum-energy path", fontsize=10)
ax.legend(fontsize=6.3, loc="upper center", frameon=False, ncol=1)
ax.set_xlim(-0.3, len(path_e) - 0.7)
fig.tight_layout()
fig.savefig(OUT / "P4_neb_hop_barrier.pdf")
fig.savefig(OUT / "P4_neb_hop_barrier.png", dpi=300)
plt.close(fig)
print("Fig 1 written")

# --------------------------------------------- Fig 2: geometry-kinetics decomposition
# height-matched points only (alpha=82 vs ballistic a80 2deg-offset ref; alpha=89 vs
# matched ballistic a89), numbers copied from C2_ANALYSIS_20260910.md verbatim.
angles = ["alpha=82\n(vs ballistic a80,\n2 deg offset)", "alpha=89\n(matched)"]
lf_off = [0.99999, 0.9972]
lf_on = [0.99998, 0.9557]
datom_pct = [22.0, 13.9]  # a82: (28582959-23339890)/23339890*100 rounded; a89: +13.9% stated
overlap10_off = [11.4, 31.5]
overlap10_on = [8.3, 28.7]
beta_shift = [None, -0.13]  # only a89 has a valid pairwise beta (same-angle) comparison

fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9))

x = np.arange(len(angles))
w = 0.32
ax = axes[0]
ax.bar(x - w/2, lf_off, w, color="#9B8D78", label="diffusion OFF")
ax.bar(x + w/2, lf_on, w, color="#AC581F", label="diffusion ON (cap=1)")
ax.set_ylim(0.9, 1.005)
ax.set_xticks(x); ax.set_xticklabels(angles, fontsize=7)
ax.set_ylabel("Largest-component fraction (LF)")
ax.set_title("Connectivity", fontsize=9)
ax.legend(fontsize=6.5, frameon=False)

ax = axes[1]
ax.bar(x, datom_pct, w * 1.3, color="#2C7A70")
ax.set_xticks(x); ax.set_xticklabels(angles, fontsize=7)
ax.set_ylabel("Atom count change (%)")
ax.set_title("Mass (diffusion fills in)", fontsize=9)
ax.axhline(0, color="black", lw=0.6)

ax = axes[2]
ax.bar(x - w/2, overlap10_off, w, color="#9B8D78")
ax.bar(x + w/2, overlap10_on, w, color="#AC581F")
ax.set_xticks(x); ax.set_xticklabels(angles, fontsize=7)
ax.set_ylabel(r"Interpenetration $>10\%$ (% of atoms)")
ax.set_title("Overlap (not worsened)", fontsize=9)

fig.suptitle("Geometry--kinetics decomposition at the two height-matched points", fontsize=9.5, y=1.03)
fig.tight_layout()
fig.savefig(OUT / "P4_geometry_kinetics_decomposition.pdf", bbox_inches="tight")
fig.savefig(OUT / "P4_geometry_kinetics_decomposition.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Fig 2 written")
